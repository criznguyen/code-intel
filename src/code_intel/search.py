"""High-level search functions used by both MCP server and CLI."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import threading
from typing import Any

from code_intel._logging import get_logger
from code_intel.config import Config
from code_intel.embedder import EmbeddingProvider, get_provider

log = get_logger(__name__)

# Provider cache: re-creating an OllamaProvider per query also recreates its
# httpx.Client, which discards keep-alive connections. Cache per
# (provider_name, endpoint, model, dim, batch_size, timeout_seconds) so config
# changes between calls invalidate cleanly.
_PROVIDER_CACHE: dict[tuple[str, str, str, int, int, float], EmbeddingProvider] = {}
_PROVIDER_CACHE_LOCK = threading.Lock()


def _provider_cache_key(cfg: Config) -> tuple[str, str, str, int, int, float]:
    e = cfg.embedding
    return (
        e.provider,
        e.endpoint,
        e.model,
        int(e.dim),
        int(e.batch_size),
        float(e.timeout_seconds),
    )


def _get_cached_provider(cfg: Config) -> EmbeddingProvider:
    key = _provider_cache_key(cfg)
    with _PROVIDER_CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(key)
        if cached is not None:
            return cached
        provider = get_provider(cfg)
        _PROVIDER_CACHE[key] = provider
        return provider


def _reset_provider_cache() -> None:
    """Test-only: clear cached providers so per-test config changes apply."""
    with _PROVIDER_CACHE_LOCK:
        _PROVIDER_CACHE.clear()


# Tokens stripped from query/symbol tokenization before computing lexical
# overlap. Keep this list minimal — false-positive boosts hurt less than
# missing a relevant match.
_RERANK_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "with", "by", "at", "as", "be", "this", "that",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _symbol_tokens(symbol: str) -> set[str]:
    """Tokenize a symbol into lowercase alphanumeric chunks.

    Handles snake_case (`calculate_fee`), camelCase / PascalCase
    (`TransferFeeConfig` → ``{"transferfeeconfig", "transfer", "fee", "config"}``)
    by also injecting camelCase split tokens. Conservative — false splits hurt
    only the boost step, not recall.
    """
    s = symbol.lower()
    tokens = set(_TOKEN_RE.findall(s))
    # CamelCase split on the original (uppercase markers preserve word edges).
    camel = re.findall(r"[A-Z]+(?:[a-z0-9]+)?|[a-z0-9]+", symbol)
    tokens.update(t.lower() for t in camel if t)
    return tokens


def _query_tokens(query: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(query.lower()) if t not in _RERANK_STOPWORDS}


# Rerank weight constants (multiplicative on LanceDB L2 distance; lower=better).
# Tuned on solanabot v0.1.4 corpus: `calculate token2022 transfer fee` lifts
# `calculate_fee` from #17 to top-5 without disturbing top docs for general queries.
_KIND_BOOST_FUNCTION = 0.85       # functions are usually the user's target
_TEST_NAME_PENALTY = 1.15          # de-prioritize `test_…` symbols
_LEX_OVERLAP_STRONG = 0.88         # ≥2 query tokens appear in symbol
_LEX_OVERLAP_WEAK = 0.94           # ≥1 query token appears in symbol
# Symbol-coverage bonus: when the matched tokens cover most of the symbol
# (i.e. the symbol *is* mostly the query, like `calculate_fee` ↔ "calculate fee"),
# the match is much higher-signal than the same overlap on a long symbol.
_COVERAGE_FULL = 0.85              # ≥90% of symbol tokens are in query
_COVERAGE_HALF = 0.92              # ≥50% of symbol tokens are in query
_RERANK_OVERFETCH = 3              # fetch k*N candidates then trim


def _rerank(rows: list[dict[str, Any]], q_tokens: set[str]) -> list[dict[str, Any]]:
    """Apply small heuristic boosts then re-sort. Returns a new list."""
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        base = r.get("_distance")
        if base is None:
            # search-without-vector returns no _distance; skip rerank, keep order.
            scored.append((0.0, r))
            continue
        score = float(base)
        kind = r.get("kind", "")
        symbol = r.get("symbol", "") or ""
        if kind in ("function", "method"):
            score *= _KIND_BOOST_FUNCTION
            # Strip `:partN` suffix so split chunks still flag as tests.
            sym_head = symbol.split(":part", 1)[0]
            if sym_head.startswith("test_") or sym_head.startswith("Test"):
                score *= _TEST_NAME_PENALTY
        if q_tokens:
            sym_tok = _symbol_tokens(symbol)
            overlap = len(q_tokens & sym_tok)
            if overlap >= 2:
                score *= _LEX_OVERLAP_STRONG
            elif overlap >= 1:
                score *= _LEX_OVERLAP_WEAK
            # Symbol-coverage bonus: rewards dense matches (short symbols
            # that are *mostly* the query) over partial matches on long
            # symbols. `calculate_fee` (cov=1.0) beats
            # `parse_transfer_fee_config_value` (cov=0.4) on
            # "calculate token2022 transfer fee" even though both have
            # 2 overlapping tokens.
            if sym_tok and overlap >= 1:
                coverage = overlap / len(sym_tok)
                if coverage >= 0.9:
                    score *= _COVERAGE_FULL
                elif coverage >= 0.5:
                    score *= _COVERAGE_HALF
        scored.append((score, r))
    scored.sort(key=lambda x: x[0])
    # Preserve the post-rerank score on the row so callers can inspect it.
    out: list[dict[str, Any]] = []
    for s, r in scored:
        r2 = dict(r)
        r2["_rerank_score"] = s
        out.append(r2)
    return out


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def search_lexical(
    cfg: Config,
    pattern: str,
    path_glob: str | None = None,
    lang: str | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """ripgrep-backed lexical search. Returns [{path, line, snippet}]."""
    rg = _which("rg")
    if not rg:
        raise RuntimeError("ripgrep ('rg') not found in PATH. Install it for lexical search.")
    cmd = [
        rg,
        "--no-heading",
        "--with-filename",
        "--line-number",
        "--color=never",
        "-m",
        str(max_results),
    ]
    if lang:
        cmd += ["--type", lang]
    if path_glob:
        cmd += ["-g", path_glob]
    cmd += ["--", pattern, str(cfg.target)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:  # pragma: no cover
        raise RuntimeError(f"rg invocation failed: {e}") from e
    hits: list[dict[str, Any]] = []
    for raw in out.stdout.splitlines()[:max_results]:
        # Format: path:line:snippet
        parts = raw.split(":", 2)
        if len(parts) < 3:
            continue
        hits.append({"path": parts[0], "line": int(parts[1]), "snippet": parts[2]})
    return hits


def semantic_search(
    cfg: Config,
    query: str,
    k: int = 10,
    lang: str | None = None,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Embed query and search LanceDB.

    When ``rerank=True`` (default) we over-fetch ``k * _RERANK_OVERFETCH``
    candidates and re-rank with cheap lexical + kind-based heuristics before
    trimming back to ``k``. Set ``rerank=False`` to bypass and return raw L2
    nearest-neighbor order (useful for benchmarking the embedding model).
    """
    if not query or not query.strip():
        # Empty / whitespace-only queries explode in Ollama (zero-length
        # embedding vector → LanceDB IndexError downstream). Fail fast with
        # a typed error the caller can render. (HIGH-1 in v0.1.3 audit.)
        raise ValueError("query must be non-empty")
    provider = _get_cached_provider(cfg)
    result = provider.embed([query])
    if not result.vectors:
        reason = result.skipped_reasons.get(0, "unknown")
        raise RuntimeError(f"failed to embed query: {reason}")
    vec = result.vectors[0]

    from code_intel.store import search as db_search

    fetch_k = max(k * _RERANK_OVERFETCH, k) if rerank else k
    rows = db_search(cfg, vec, k=fetch_k, lang=lang)
    if rerank and rows:
        rows = _rerank(rows, _query_tokens(query))
    rows = rows[:k]
    return [
        {
            "path": r["path"],
            "symbol": r["symbol"],
            "kind": r["kind"],
            "lang": r["lang"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "content": r["content"],
            "score": r.get("_distance"),
            "rerank_score": r.get("_rerank_score"),
        }
        for r in rows
    ]


def structural_search(
    cfg: Config,
    pattern: str,
    lang: str,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """ast-grep-based structural search."""
    ag = _which("ast-grep") or _which("sg")
    if not ag:
        raise RuntimeError("ast-grep ('ast-grep' or 'sg') not found in PATH.")
    cmd = [
        ag,
        "run",
        "-p",
        pattern,
        "--lang",
        lang,
        "--json=stream",
        str(cfg.target),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:  # pragma: no cover
        raise RuntimeError(f"ast-grep invocation failed: {e}") from e

    import json

    hits: list[dict[str, Any]] = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        hits.append(
            {
                "path": obj.get("file"),
                "line": (obj.get("range") or {}).get("start", {}).get("line"),
                "snippet": (obj.get("text") or "").splitlines()[0] if obj.get("text") else "",
            }
        )
        if len(hits) >= max_results:
            break
    return hits


def quick_cli_search(cfg: Config, query: str, k: int = 5) -> str:
    """Render a quick human-readable semantic search summary for CLI debugging."""
    try:
        results = semantic_search(cfg, query, k=k)
    except Exception as e:
        return f"semantic search failed: {e}"
    if not results:
        return "(no results)"
    parts = [f"{shlex.quote(query)} -> {len(results)} hits"]
    for r in results:
        parts.append(
            f"  {r['path']}:{r['start_line']}-{r['end_line']}  {r['symbol']} ({r['kind']})"
        )
    return "\n".join(parts)
