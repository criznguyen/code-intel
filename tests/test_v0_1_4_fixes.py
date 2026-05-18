"""Regression tests for the 11 findings closed in code-intel v0.1.4.

Each test pins one finding's contract so future refactors can't quietly
reopen the bug. Findings reference the v0.1.3 audit transcript.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

from code_intel.chunker import (
    _RUST_MOD_DECL_RE,
    _whole_file_chunk,
    chunk_text,
)
from code_intel.config import default_config
from code_intel.embedder import OllamaProvider
from code_intel.search import semantic_search
from code_intel.store import _reset_db_cache, _sql_quote, open_db

# ---------------------------------------------------------------------------
# HIGH-1 — empty query / zero-length vector
# ---------------------------------------------------------------------------


def test_semantic_search_rejects_empty_query(tmp_path) -> None:
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    with pytest.raises(ValueError, match="non-empty"):
        semantic_search(cfg, "", k=5)
    with pytest.raises(ValueError, match="non-empty"):
        semantic_search(cfg, "   \n\t  ", k=5)


@respx.mock
def test_embedder_rejects_zero_length_vector() -> None:
    """Ollama HTTP 200 with embedding=[] (whitespace prompt) must raise, not
    push a zero-length vector into LanceDB."""
    cfg = default_config(project_name="t")
    cfg.embedding.dim = 3
    provider = OllamaProvider(cfg.embedding)
    provider._batch_endpoint_disabled = True  # legacy per-item path
    respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": []})
    )
    result = provider.embed(["   "])
    # zero-length vector path goes through the per-item RuntimeError branch:
    # skipped instead of crashing.
    assert result.vectors == []
    assert result.skipped_indices == [0]
    assert "protocol" in result.skipped_reasons[0]


# ---------------------------------------------------------------------------
# HIGH-2 — _whole_file_chunk respects max_chunk_chars
# ---------------------------------------------------------------------------


def test_whole_file_fallback_respects_max_chunk_chars() -> None:
    # Module-level python: no def/class, so the chunker hits _whole_file_chunk.
    body = "\n".join(f"X_{i} = {i!r}" for i in range(800))
    chunks = chunk_text("flat.py", "python", body, max_chunk_chars=400)
    assert len(chunks) >= 2, "should split the head section, not emit 1 giant chunk"
    for c in chunks:
        assert len(c.content) <= 400


def test_whole_file_chunk_returns_list() -> None:
    """Contract: _whole_file_chunk always returns a list (callers iterate)."""
    out = _whole_file_chunk("a.py", "python", "x = 1\n")
    assert isinstance(out, list)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# MED-3 — Rust `pub mod foo;` forward-decl is dropped, inline `mod {}` kept
# ---------------------------------------------------------------------------


def _has_rust() -> bool:
    from code_intel.chunker import _get_parser

    return _get_parser("rust") is not None


@pytest.mark.skipif(not _has_rust(), reason="tree-sitter-rust not installed")
def test_rust_forward_module_decl_skipped() -> None:
    """`mod foo;` and `pub mod bar;` are 1-line forward decls — filtered."""
    src = """\
pub mod a;
mod b;
pub(crate) mod c;

pub fn keep_me() -> i32 { 1 }
"""
    chunks = chunk_text("lib.rs", "rust", src)
    symbols = {c.symbol for c in chunks}
    # No mod-decl junk; only the real fn.
    assert "keep_me" in symbols
    # The forward decls should not have produced "a", "b", "c" mod chunks.
    mod_chunks = [c for c in chunks if c.kind == "module" and c.symbol in {"a", "b", "c"}]
    assert mod_chunks == [], f"forward mod decls leaked: {[c.symbol for c in mod_chunks]}"


@pytest.mark.skipif(not _has_rust(), reason="tree-sitter-rust not installed")
def test_rust_inline_module_body_kept() -> None:
    """`mod foo { ... }` (body) IS kept — only 1-line decls are dropped."""
    src = """\
pub mod helpers {
    pub fn h() -> i32 { 42 }
}
"""
    chunks = chunk_text("lib.rs", "rust", src)
    symbols = {c.symbol for c in chunks}
    assert "helpers" in symbols or "h" in symbols, (
        f"inline mod body should survive; got {symbols}"
    )


def test_rust_mod_decl_regex() -> None:
    """Pin the regex spec independently of tree-sitter availability."""
    assert _RUST_MOD_DECL_RE.match("mod foo;")
    assert _RUST_MOD_DECL_RE.match("pub mod foo;")
    assert _RUST_MOD_DECL_RE.match("pub(crate) mod foo ;")
    assert _RUST_MOD_DECL_RE.match("   pub mod x;   ")
    # Inline body must NOT match (has braces).
    assert not _RUST_MOD_DECL_RE.match("pub mod foo { }")
    assert not _RUST_MOD_DECL_RE.match("mod foo {\n  pub fn h() {}\n}")


# ---------------------------------------------------------------------------
# MED-4 — markdown fence state-tracking
# ---------------------------------------------------------------------------


def test_markdown_fenced_shell_comment_not_section() -> None:
    md = """\
# Real Heading

Intro paragraph.

```bash
# Setup
export FOO=1
# Cleanup
echo done
```

## After Code

Trailing.
"""
    chunks = chunk_text("readme.md", "markdown", md)
    titles = {c.symbol for c in chunks}
    assert "Real Heading" in titles
    assert "After Code" in titles
    # The shell comments inside ``` must NOT become heading chunks.
    assert "Setup" not in titles
    assert "Cleanup" not in titles


def test_markdown_tilde_fence_also_handled() -> None:
    """~~~ fences are valid CommonMark too; state tracker must toggle on them."""
    md = """\
# Top

Top intro body.

~~~
# fake heading
~~~

## Real Next

Real body content under Real Next.
"""
    chunks = chunk_text("doc.md", "markdown", md)
    titles = {c.symbol for c in chunks}
    assert "Top" in titles
    assert "Real Next" in titles
    assert "fake heading" not in titles


# ---------------------------------------------------------------------------
# MED-5 — SQL apostrophe escape
# ---------------------------------------------------------------------------


def test_sql_quote_doubles_apostrophes() -> None:
    assert _sql_quote("x") == "'x'"
    assert _sql_quote("it's") == "'it''s'"
    assert _sql_quote("a'b'c") == "'a''b''c'"
    assert _sql_quote("") == "''"


def test_upsert_chunks_path_with_apostrophe(tmp_path) -> None:
    """Re-upserting a chunk whose path contains `'` must replace, not duplicate."""
    pytest.importorskip("lancedb")
    from code_intel.chunker import Chunk
    from code_intel.store import table_stats, upsert_chunks

    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    cfg.embedding.dim = 2

    chunk = Chunk(
        path="src/it's.rs",
        lang="rust",
        symbol="fn_it",
        kind="function",
        start_line=1,
        end_line=2,
        content="fn it() {}",
        content_hash="abc",
    )
    vec = [0.1, 0.2]

    upsert_chunks(cfg, [chunk], [vec])
    upsert_chunks(cfg, [chunk], [vec])

    stats = table_stats(cfg)
    # The second upsert MUST replace the first row, not append a duplicate.
    assert stats["rows"] == 1, f"expected 1 row after dedupe, got {stats['rows']}"
    _reset_db_cache()


# ---------------------------------------------------------------------------
# MED-6 — watcher delete + CLI --prune
# ---------------------------------------------------------------------------


def test_watcher_handles_delete_event(tmp_path) -> None:
    """awatch yields a Change.deleted event → delete_for_path is called."""
    pytest.importorskip("watchfiles")
    import asyncio

    from watchfiles import Change

    from code_intel import watcher as watcher_mod
    from code_intel.config import save_config

    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    save_config(cfg, tmp_path)

    (tmp_path / "src").mkdir()
    deleted_file = tmp_path / "src" / "gone.py"
    deleted_file.write_text("x = 1\n")  # exists at watch start

    async def fake_awatch(*args, **kwargs):
        # First yield: a Change.deleted event. Then the function will hit
        # StopAsyncIteration and the watcher loop exits naturally.
        # But the watcher uses `async for changes in awatch(...)` which
        # iterates until the underlying gen is exhausted, so we yield once
        # and then return.
        yield {(Change.deleted, str(deleted_file))}

    calls: list[str] = []

    def fake_delete(cfg_arg, rel: str) -> int:
        calls.append(rel)
        return 1

    # Remove the file so the watcher sees it as truly gone (defensive).
    deleted_file.unlink()

    with (
        patch.object(watcher_mod, "load_config", return_value=cfg),
        patch("watchfiles.awatch", fake_awatch),
        patch("code_intel.store.delete_for_path", fake_delete),
    ):
        asyncio.run(asyncio.wait_for(watcher_mod.watch(tmp_path), timeout=2.0))

    assert calls == ["src/gone.py"], f"expected delete_for_path call, got {calls}"


def test_index_prune_removes_orphans(tmp_path) -> None:
    """prune_orphans: paths in DB but not on disk → deleted."""
    pytest.importorskip("lancedb")
    from code_intel.chunker import Chunk
    from code_intel.indexer import prune_orphans
    from code_intel.store import list_indexed_paths, upsert_chunks

    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    cfg.embedding.dim = 2

    # Seed DB with one path that exists and one that doesn't.
    real_dir = tmp_path / "src"
    real_dir.mkdir()
    real_file = real_dir / "keep.py"
    real_file.write_text("x = 1\n")

    chunks = [
        Chunk(
            path="src/keep.py",
            lang="python",
            symbol="m",
            kind="module",
            start_line=1,
            end_line=1,
            content="x = 1",
            content_hash="h1",
        ),
        Chunk(
            path="src/ghost.py",
            lang="python",
            symbol="m",
            kind="module",
            start_line=1,
            end_line=1,
            content="y = 2",
            content_hash="h2",
        ),
    ]
    upsert_chunks(cfg, chunks, [[0.1, 0.2], [0.3, 0.4]])
    pre = list_indexed_paths(cfg)
    assert {"src/keep.py", "src/ghost.py"} <= pre

    removed = prune_orphans(cfg)
    assert removed == 1
    post = list_indexed_paths(cfg)
    assert "src/keep.py" in post
    assert "src/ghost.py" not in post
    _reset_db_cache()


# ---------------------------------------------------------------------------
# LOW-7 — configurable timeout
# ---------------------------------------------------------------------------


def test_embedding_timeout_configurable() -> None:
    """timeout_seconds from config flows through to httpx.Client."""
    cfg = default_config(project_name="t")
    cfg.embedding.timeout_seconds = 0.5
    provider = OllamaProvider(cfg.embedding)
    # httpx exposes a Timeout object; the connect/read/write/pool fields all
    # share the constructor scalar.
    t = provider._client.timeout
    assert float(t.read) == 0.5, f"timeout did not propagate; got {t}"


# ---------------------------------------------------------------------------
# LOW-8 — batch_size used (/api/embed) with 404 fallback
# ---------------------------------------------------------------------------


@respx.mock
def test_embed_batch_uses_input_array() -> None:
    """1 POST per batch_size-sized slice, not 1 per input."""
    cfg = default_config(project_name="t")
    cfg.embedding.dim = 2
    cfg.embedding.batch_size = 3
    provider = OllamaProvider(cfg.embedding)

    route = respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(
            200, json={"embeddings": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]}
        )
    )
    result = provider.embed(["a", "b", "c"])
    assert route.call_count == 1, f"expected 1 batched POST, got {route.call_count}"
    assert len(result.vectors) == 3


@respx.mock
def test_embed_falls_back_on_404() -> None:
    """Old Ollama (no /api/embed) → 404 → silently fall through to per-item."""
    cfg = default_config(project_name="t")
    cfg.embedding.dim = 2
    cfg.embedding.batch_size = 4
    provider = OllamaProvider(cfg.embedding)

    respx.post("http://localhost:11434/api/embed").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )
    legacy = respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.9, 0.9]})
    )
    result = provider.embed(["a", "b"])
    assert len(result.vectors) == 2
    # Sticky bit set → subsequent embed() calls go straight to legacy.
    assert provider._batch_endpoint_disabled is True
    assert legacy.call_count == 2


# ---------------------------------------------------------------------------
# LOW-9 — DB handle cached
# ---------------------------------------------------------------------------


def test_search_reuses_db_handle(tmp_path) -> None:
    """open_db caches the lancedb connection per (path, table)."""
    _reset_db_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path

    fake_conn = MagicMock(name="conn")
    with patch("lancedb.connect", return_value=fake_conn) as connect:
        a = open_db(cfg)
        b = open_db(cfg)
        c = open_db(cfg)
    assert a is b is c
    assert connect.call_count == 1, f"expected 1 connect, got {connect.call_count}"
    _reset_db_cache()


# ---------------------------------------------------------------------------
# INFO-11 part A — walk_repo prunes excluded dirs
# ---------------------------------------------------------------------------


def test_walk_repo_prunes_target_dir(tmp_path, monkeypatch) -> None:
    """Excluded dirs (target/) must not be descended into.

    We assert by counting how many files inside target/ get yielded — must be 0.
    """
    from code_intel.indexer import _walk_repo

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "big.rs").write_text("// generated\n")
    (target_dir / "nested").mkdir()
    (target_dir / "nested" / "deeper.rs").write_text("// also generated\n")

    src = tmp_path / "src"
    src.mkdir()
    (src / "main.rs").write_text("fn main() {}\n")

    cfg = default_config(project_name="t")
    cfg._target = tmp_path
    # default exclude_globs already contains "**/target/**".
    files = list(_walk_repo(cfg))
    rels = {str(p.relative_to(tmp_path)) for p in files}
    assert "src/main.rs" in rels
    assert all("target" not in r for r in rels), f"target/ leaked: {rels}"
