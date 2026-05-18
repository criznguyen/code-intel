"""Regression tests for the 4 items closed in code-intel v0.1.5.

Each test pins one fix's contract so future refactors can't quietly reopen
the bug. Items reference the v0.1.4 audit residual list.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from code_intel.chunker import chunk_text
from code_intel.config import default_config
from code_intel.search import (
    _query_tokens,
    _rerank,
    _reset_provider_cache,
    _symbol_tokens,
)
from code_intel.store import _reset_db_cache

# ---------------------------------------------------------------------------
# MED-4 residual — drop heading-only markdown sections
# ---------------------------------------------------------------------------


def test_markdown_drops_heading_only_section() -> None:
    """A section whose body is just the heading line is useless retrieval
    noise (e.g. `## Date: 2026-03-28` followed immediately by `## Version: 1.0`)."""
    md = """\
# Real Title

Intro body paragraph here.

## Date: 2026-03-28
## Version: 1.0
## Status: Draft

## After Metadata

Trailing content body.
"""
    chunks = chunk_text("doc.md", "markdown", md)
    titles = {c.symbol for c in chunks}
    # Real sections with body survive.
    assert "Real Title" in titles
    assert "After Metadata" in titles
    # Heading-only metadata sections are dropped.
    assert "Date: 2026-03-28" not in titles
    assert "Version: 1.0" not in titles
    assert "Status: Draft" not in titles


def test_markdown_keeps_one_line_section_with_body() -> None:
    """Single-line content that is NOT a heading line stays a section."""
    md = """\
# Heading

Just one line of body text.

## Another

More body here.
"""
    chunks = chunk_text("doc.md", "markdown", md)
    titles = {c.symbol for c in chunks}
    assert "Heading" in titles
    assert "Another" in titles


def test_markdown_all_heading_only_falls_back() -> None:
    """If a doc is nothing but stacked metadata, we still emit a fallback chunk
    so the file isn't invisible to retrieval."""
    md = "## Date: 2026-05-18\n## Version: 1.0\n## Status: Draft\n"
    chunks = chunk_text("meta.md", "markdown", md)
    # Fallback whole-file chunk surfaces, not zero chunks.
    assert chunks, "all-heading-only file should still fall through to whole-file fallback"


# ---------------------------------------------------------------------------
# LOW-9 follow-up — provider cache
# ---------------------------------------------------------------------------


def test_provider_cache_reuses_instance(tmp_path) -> None:
    """semantic_search reuses the cached OllamaProvider (and its httpx.Client)
    across calls instead of constructing a new one per query."""
    pytest.importorskip("lancedb")
    from code_intel.search import _get_cached_provider

    _reset_provider_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path

    with patch("code_intel.search.get_provider") as mock_get:
        # Sentinel: any object will do, but make the cache key resolve.
        sentinel = object()
        mock_get.return_value = sentinel
        a = _get_cached_provider(cfg)
        b = _get_cached_provider(cfg)
        c = _get_cached_provider(cfg)
    assert a is b is c is sentinel
    assert mock_get.call_count == 1, f"expected 1 get_provider call, got {mock_get.call_count}"
    _reset_provider_cache()


def test_provider_cache_invalidates_on_config_change(tmp_path) -> None:
    """Changing endpoint / model / dim must produce a fresh provider, not the
    previously-cached one (otherwise users get silent stale-config behavior)."""
    from code_intel.search import _get_cached_provider

    _reset_provider_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path

    seen: list[object] = []

    def fake_get_provider(_cfg):
        obj = object()
        seen.append(obj)
        return obj

    with patch("code_intel.search.get_provider", side_effect=fake_get_provider):
        a = _get_cached_provider(cfg)
        # Change endpoint -> cache key differs.
        cfg.embedding.endpoint = "http://other:11434"
        b = _get_cached_provider(cfg)
        # Change model -> cache key differs again.
        cfg.embedding.model = "nomic-embed-text"
        c = _get_cached_provider(cfg)
    assert a is not b
    assert b is not c
    assert len(seen) == 3
    _reset_provider_cache()


# ---------------------------------------------------------------------------
# INFO-11 part B — content-hash dedup for --since
# ---------------------------------------------------------------------------


def test_lookup_existing_hashes_returns_stored(tmp_path) -> None:
    """lookup_existing_hashes maps (path, symbol, start_line) -> content_hash
    for stored rows in the affected path set."""
    pytest.importorskip("lancedb")
    from code_intel.chunker import Chunk
    from code_intel.store import lookup_existing_hashes, upsert_chunks

    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    cfg.embedding.dim = 2

    chunks = [
        Chunk(
            path="src/keep.py",
            lang="python",
            symbol="f",
            kind="function",
            start_line=10,
            end_line=20,
            content="def f(): pass",
            content_hash="HASH-A",
        ),
        Chunk(
            path="src/keep.py",
            lang="python",
            symbol="g",
            kind="function",
            start_line=30,
            end_line=40,
            content="def g(): pass",
            content_hash="HASH-B",
        ),
        Chunk(
            path="src/other.py",
            lang="python",
            symbol="h",
            kind="function",
            start_line=1,
            end_line=2,
            content="def h(): pass",
            content_hash="HASH-C",
        ),
    ]
    upsert_chunks(cfg, chunks, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    result = lookup_existing_hashes(cfg, {"src/keep.py"})
    assert result == {
        ("src/keep.py", "f", 10): "HASH-A",
        ("src/keep.py", "g", 30): "HASH-B",
    }
    # Path not in DB returns no entries.
    result = lookup_existing_hashes(cfg, {"src/missing.py"})
    assert result == {}
    # Empty path set returns empty dict without querying.
    assert lookup_existing_hashes(cfg, set()) == {}
    _reset_db_cache()


def test_index_repo_since_skips_unchanged_chunks(tmp_path) -> None:
    """When --since is set and chunk content_hash matches the stored row, we
    must NOT re-embed it. Only changed/new chunks pay the embed cost."""
    pytest.importorskip("lancedb")
    from code_intel.embedder import EmbedResult
    from code_intel.indexer import index_repo

    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    cfg.embedding.dim = 2

    # Seed a python file with two functions.
    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")

    # First pass: full index. Mock the embed provider to return fixed vectors.
    seen_batches: list[int] = []

    def make_provider(vectors_per_call: list[list[float]]):
        class P:
            name = "fake"
            dim = 2

            def embed(self, texts):
                seen_batches.append(len(texts))
                vecs = [[0.1, 0.2]] * len(texts)
                return EmbedResult(vectors=vecs, skipped_indices=[], skipped_reasons={})

        return P()

    with patch("code_intel.indexer.get_provider", return_value=make_provider([])):
        stats1 = index_repo(cfg, since=None)
    assert stats1["chunks"] > 0
    initial_embed_calls = sum(seen_batches)
    assert initial_embed_calls == stats1["chunks"]

    # Second pass: --since with NO file changes. Same content -> same hashes.
    # The chunker is deterministic on identical content, so every chunk should
    # be a cache hit. Override _git_changed_files since there's no real git
    # repo here.
    seen_batches.clear()
    with (
        patch("code_intel.indexer._git_changed_files", return_value=[src / "mod.py"]),
        patch("code_intel.indexer.get_provider", return_value=make_provider([])),
    ):
        stats2 = index_repo(cfg, since="HEAD")
    # All chunks must be cache hits.
    assert stats2["cache_hits"] == stats2["chunks"], (
        f"expected all hits; got cache_hits={stats2['cache_hits']} chunks={stats2['chunks']}"
    )
    assert stats2["embedded"] == 0
    assert sum(seen_batches) == 0, f"embed must not be called; got {seen_batches}"

    # Third pass: modify ONE function body so its content_hash changes.
    # The other function must still be a cache hit.
    (src / "mod.py").write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 999  # changed\n"
    )
    seen_batches.clear()
    with (
        patch("code_intel.indexer._git_changed_files", return_value=[src / "mod.py"]),
        patch("code_intel.indexer.get_provider", return_value=make_provider([])),
    ):
        stats3 = index_repo(cfg, since="HEAD")
    # Exactly 1 chunk changed (beta); the rest are cache hits.
    assert stats3["cache_hits"] >= 1, f"alpha should still hit cache; stats={stats3}"
    assert sum(seen_batches) >= 1, "beta must trigger an embed call"
    assert sum(seen_batches) < stats3["chunks"], (
        "must NOT re-embed every chunk when only one changed"
    )
    _reset_db_cache()


def test_index_repo_since_force_bypasses_cache(tmp_path) -> None:
    """force=True must re-embed everything even if content_hash matches."""
    pytest.importorskip("lancedb")
    from code_intel.embedder import EmbedResult
    from code_intel.indexer import index_repo

    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    cfg.embedding.dim = 2

    src = tmp_path / "src"
    src.mkdir()
    (src / "mod.py").write_text("def alpha():\n    return 1\n")

    seen_batches: list[int] = []

    class P:
        name = "fake"
        dim = 2

        def embed(self, texts):
            seen_batches.append(len(texts))
            return EmbedResult(
                vectors=[[0.1, 0.2]] * len(texts), skipped_indices=[], skipped_reasons={}
            )

    with patch("code_intel.indexer.get_provider", return_value=P()):
        index_repo(cfg, since=None)

    # --since with force=True must NOT consult the cache.
    seen_batches.clear()
    with (
        patch("code_intel.indexer._git_changed_files", return_value=[src / "mod.py"]),
        patch("code_intel.indexer.get_provider", return_value=P()),
    ):
        stats = index_repo(cfg, since="HEAD", force=True)
    assert stats["cache_hits"] == 0, f"force must bypass cache; got {stats}"
    assert sum(seen_batches) == stats["chunks"]
    _reset_db_cache()


# ---------------------------------------------------------------------------
# Reranker — kind=function boost + lex-overlap + test_* penalty
# ---------------------------------------------------------------------------


def test_query_tokens_strips_stopwords() -> None:
    tokens = _query_tokens("calculate the token2022 transfer fee for a swap")
    assert "calculate" in tokens
    assert "token2022" in tokens
    assert "transfer" in tokens
    assert "fee" in tokens
    assert "swap" in tokens
    # Stopwords filtered.
    assert "the" not in tokens
    assert "for" not in tokens
    assert "a" not in tokens


def test_symbol_tokens_splits_snake_and_camel() -> None:
    assert "calculate" in _symbol_tokens("calculate_fee")
    assert "fee" in _symbol_tokens("calculate_fee")
    # CamelCase / PascalCase split.
    cam = _symbol_tokens("TransferFeeConfig")
    assert "transfer" in cam
    assert "fee" in cam
    assert "config" in cam


def test_rerank_promotes_function_with_lex_overlap() -> None:
    """A non-test function with lex-overlap must rank above its test_* twin
    (the test_ penalty) and above a generic class chunk with no lex overlap."""
    rows = [
        {
            "_distance": 0.92,
            "kind": "class",
            "symbol": "GenericConfig",  # no lex overlap with query
            "path": "src/other.rs",
        },
        {
            "_distance": 1.08,
            "kind": "function",
            "symbol": "calculate_fee",
            "path": "src/token2022/mod.rs",
        },
        {
            "_distance": 0.95,
            "kind": "function",
            "symbol": "test_calculate_fee",
            "path": "src/token2022/mod.rs",
        },
    ]
    q_tokens = _query_tokens("calculate token2022 transfer fee")
    out = _rerank(rows, q_tokens)
    symbols_in_order = [r["symbol"] for r in out]
    # calculate_fee should rank above test_calculate_fee (test penalty).
    assert symbols_in_order.index("calculate_fee") < symbols_in_order.index("test_calculate_fee")
    # And above the no-overlap class chunk (function boost + lex overlap wins).
    assert symbols_in_order.index("calculate_fee") < symbols_in_order.index("GenericConfig")


def test_rerank_symbol_coverage_beats_partial_long_match() -> None:
    """A short symbol that *is* the query (cov=1.0) must outrank a long symbol
    with the same raw distance but only partial coverage (cov=0.4).

    Tuned for queries like "calculate token2022 transfer fee" where
    ``calculate_fee`` (2/2 tokens matched) should beat
    ``parse_transfer_fee_config_value`` (2/5 tokens matched) even though the
    long symbol has a slightly better raw embedding distance.
    """
    rows = [
        {
            "_distance": 1.02,
            "kind": "function",
            "symbol": "parse_transfer_fee_config_value",  # 5 tokens, 2 overlap
        },
        {
            "_distance": 1.08,
            "kind": "function",
            "symbol": "calculate_fee",  # 2 tokens, 2 overlap (cov=1.0)
        },
    ]
    q = {"calculate", "token2022", "transfer", "fee"}
    out = _rerank(rows, q_tokens=q)
    syms = [r["symbol"] for r in out]
    assert syms.index("calculate_fee") < syms.index("parse_transfer_fee_config_value")


def test_rerank_preserves_order_when_no_signals() -> None:
    """Generic query terms with no lex overlap leave kind-only reordering."""
    rows = [
        {"_distance": 1.0, "kind": "section", "symbol": "A"},
        {"_distance": 1.1, "kind": "function", "symbol": "B"},
        {"_distance": 1.2, "kind": "class", "symbol": "C"},
    ]
    out = _rerank(rows, q_tokens=set())
    # function boost (0.85) pulls B (1.1 -> 0.935) ahead of A (1.0) and C (1.2).
    assert [r["symbol"] for r in out] == ["B", "A", "C"]


def test_rerank_attaches_score() -> None:
    """Reranked rows carry the post-boost score for inspection."""
    rows = [{"_distance": 1.0, "kind": "function", "symbol": "foo"}]
    out = _rerank(rows, q_tokens={"foo"})
    # function (0.85) * lex-overlap-weak (0.94) * coverage-full (0.85) = 0.679
    assert "_rerank_score" in out[0]
    assert 0.67 < out[0]["_rerank_score"] < 0.69
