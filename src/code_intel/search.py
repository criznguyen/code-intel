"""High-level search functions used by both MCP server and CLI."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from typing import Any

from code_intel._logging import get_logger
from code_intel.config import Config
from code_intel.embedder import get_provider

log = get_logger(__name__)


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
) -> list[dict[str, Any]]:
    """Embed query and search LanceDB."""
    provider = get_provider(cfg)
    result = provider.embed([query])
    if not result.vectors:
        reason = result.skipped_reasons.get(0, "unknown")
        raise RuntimeError(f"failed to embed query: {reason}")
    vec = result.vectors[0]

    from code_intel.store import search as db_search

    rows = db_search(cfg, vec, k=k, lang=lang)
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
