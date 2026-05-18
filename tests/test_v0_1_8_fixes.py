"""Regression tests for the 3 indexer-hardening items shipped in v0.1.8.

Items:

* Gap 5 (MED, ship-blocker) — ``indexer.index_repo`` runs the
  ``_check_lancedb_writable`` pre-flight before any embed call so a
  readonly target fails fast (no wasted 30s embed pass). Rust crate path
  leaks in lance OSError messages are stripped by
  ``store._sanitize_lance_error`` and re-raised as a clean RuntimeError.
* Gap 1 (LOW) — ``index_repo`` now embeds + upserts in checkpoint batches
  instead of a single all-or-nothing pass. A SIGINT mid-stream preserves
  every prior batch's rows.
* Gap 3 (LOW) — ``_parse_gitmodules`` reads ``.gitmodules`` and
  appends ``<path>/**`` patterns to the runtime exclude spec, so vendored
  submodules don't silently consume embed budget.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from code_intel.config import default_config, load_config, save_config
from code_intel.embedder import EmbedResult
from code_intel.indexer import _parse_gitmodules, discover_files, index_repo
from code_intel.store import _reset_db_cache, _sanitize_lance_error

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bootstrap_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a tmp repo with config + given source files. Returns repo root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    cfg = default_config("test")
    cfg._target = repo
    cfg.index.include_globs = ["**/*.py"]
    save_config(cfg, target=repo)
    return repo


class _CountingStubProvider:
    """In-test embedder stub that counts embed calls so we can assert
    the pre-flight check fails before any embed work happens."""

    name = "stub"
    dim = 768
    batch_size = 32

    def __init__(self) -> None:
        self.calls = 0
        self.texts_seen = 0

    def embed(self, texts):
        self.calls += 1
        self.texts_seen += len(texts)
        return EmbedResult(vectors=[[0.1] * self.dim for _ in texts])


class _FailAfterNStub:
    """Embedder stub that raises after the Nth call, to simulate SIGINT mid-stream."""

    name = "stub"
    dim = 768
    batch_size = 32

    def __init__(self, fail_after_batches: int) -> None:
        self.fail_after_batches = fail_after_batches
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.calls > self.fail_after_batches:
            raise RuntimeError("simulated SIGINT / provider crash")
        return EmbedResult(vectors=[[0.1] * self.dim for _ in texts])


# ---------------------------------------------------------------------------
# Gap 5 — disk-full / readonly pre-flight + Rust path sanitization
# ---------------------------------------------------------------------------


def test_sanitize_lance_error_strips_rust_crate_paths() -> None:
    """Rust crate paths leak the wheel build env; we sanitize to a stable hint."""
    rust_msg = (
        "IO error: lance write failed at "
        "/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/"
        "lance-io-4.0.0/src/object_store.rs:676:21: permission denied"
    )
    cleaned = _sanitize_lance_error(rust_msg)
    assert "/root/.cargo/registry" not in cleaned
    assert "object_store.rs" not in cleaned
    assert "lance-io" not in cleaned
    assert "permission/space" in cleaned


def test_sanitize_lance_error_passes_clean_messages_through() -> None:
    """Non-Rust messages must not be mangled — operators need the real detail."""
    clean = "disk quota exceeded on /mnt/data"
    assert _sanitize_lance_error(clean) == clean


def test_upsert_chunks_sanitizes_rust_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``upsert_chunks`` wraps lance OSError → RuntimeError with sanitized msg."""
    _reset_db_cache()
    repo = _bootstrap_repo(tmp_path, {"a.py": "def f(): return 1\n"})
    cfg = load_config(repo)

    from code_intel import store
    from code_intel.chunker import Chunk

    chunk = Chunk(
        path="a.py",
        lang="python",
        symbol="f",
        kind="function",
        start_line=1,
        end_line=1,
        content="def f(): return 1\n",
        content_hash="abc",
    )

    class _BlowupTable:
        def delete(self, _q: str) -> None:
            return None

        def add(self, _rows):
            raise OSError(
                "lance commit failed at "
                "/root/.cargo/registry/src/lance-io-4.0.0/src/object_store.rs:676:21"
            )

    monkeypatch.setattr(store, "open_db", lambda _cfg: object())
    monkeypatch.setattr(
        store, "_open_or_create_table", lambda _db, _cfg, sample_rows=None: _BlowupTable()
    )

    with pytest.raises(RuntimeError) as excinfo:
        store.upsert_chunks(cfg, [chunk], [[0.0] * cfg.embedding.dim])
    msg = str(excinfo.value)
    assert "/root/.cargo/registry" not in msg
    assert "object_store.rs" not in msg
    assert "failed to upsert chunks" in msg


def test_index_repo_fails_fast_on_readonly_codeindex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight write check must run BEFORE any embed call.

    We chmod the lancedb dir to 0o555 (read+execute, no write) and assert the
    provider's ``embed`` method is never invoked — proving we did not burn
    the 30s embed pass that motivated this fix.
    """
    _reset_db_cache()
    repo = _bootstrap_repo(tmp_path, {"a.py": "def f(): return 1\n"})
    cfg = load_config(repo)

    # Make lancedb dir read-only so the writable probe fails.
    cfg.lancedb_path.mkdir(parents=True, exist_ok=True)
    os.chmod(cfg.lancedb_path, stat.S_IREAD | stat.S_IEXEC)

    if os.access(cfg.lancedb_path, os.W_OK):
        # Running as root — chmod is meaningless. Skip rather than false-positive.
        os.chmod(cfg.lancedb_path, stat.S_IRWXU)
        pytest.skip("running as root; chmod 555 still writable")

    provider = _CountingStubProvider()
    from code_intel import indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "get_provider", lambda _cfg: provider)

    try:
        with pytest.raises(RuntimeError) as excinfo:
            index_repo(cfg)
        assert "lancedb not writable" in str(excinfo.value)
        assert provider.calls == 0, (
            f"pre-flight failed but provider was still called {provider.calls} times "
            f"(this is the bug we are fixing)"
        )
    finally:
        # Restore so pytest cleanup can rm the tmp.
        os.chmod(cfg.lancedb_path, stat.S_IRWXU)


# ---------------------------------------------------------------------------
# Gap 1 — SIGINT batch-level checkpoint
# ---------------------------------------------------------------------------


def test_index_repo_commits_partial_on_provider_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider crash mid-stream must preserve all prior checkpoint batches.

    Create enough chunks to span >1 checkpoint batch, fail the provider on
    the 2nd embed call, expect:
    - RuntimeError bubbles up
    - DB contains rows from the 1st checkpoint batch
    """
    _reset_db_cache()
    # Need enough chunks to span 2 checkpoint batches (checkpoint_size=32).
    # Each tiny function body is one chunk, so create 40 functions.
    src_lines = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(40))
    repo = _bootstrap_repo(tmp_path, {"a.py": src_lines})
    cfg = load_config(repo)

    provider = _FailAfterNStub(fail_after_batches=1)
    from code_intel import indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "get_provider", lambda _cfg: provider)

    with pytest.raises(RuntimeError, match="simulated SIGINT"):
        index_repo(cfg)

    # The first batch should have committed; query the table directly.
    from code_intel.store import table_stats

    stats = table_stats(cfg)
    assert stats["rows"] > 0, (
        "no rows committed before the crash — checkpoint mechanism is broken"
    )


def test_index_repo_resume_full_produces_complete_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crash, re-running ``index_repo`` (no since=) must reach full corpus."""
    _reset_db_cache()
    src_lines = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(40))
    repo = _bootstrap_repo(tmp_path, {"a.py": src_lines})
    cfg = load_config(repo)

    # First pass: crash after batch 1.
    crashing = _FailAfterNStub(fail_after_batches=1)
    from code_intel import indexer as indexer_mod

    monkeypatch.setattr(indexer_mod, "get_provider", lambda _cfg: crashing)
    with pytest.raises(RuntimeError):
        index_repo(cfg)

    # Second pass: healthy provider, full re-embed.
    healthy = _CountingStubProvider()
    monkeypatch.setattr(indexer_mod, "get_provider", lambda _cfg: healthy)
    stats = index_repo(cfg)
    assert stats["embedded"] >= 40, (
        f"resume did not produce full corpus: embedded={stats['embedded']}"
    )


# ---------------------------------------------------------------------------
# Gap 3 — submodule auto-exclude via .gitmodules
# ---------------------------------------------------------------------------


def test_parse_gitmodules_returns_glob_paths(tmp_path: Path) -> None:
    """Sections must yield ``<path>/**`` glob entries."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitmodules").write_text(
        '[submodule "vendor-sub"]\n'
        "    path = vendor/sub\n"
        "    url = https://example.invalid/sub.git\n"
        '[submodule "third_party/thing"]\n'
        "    path = third_party/thing\n"
        "    url = https://example.invalid/thing.git\n"
    )
    out = _parse_gitmodules(repo)
    assert "vendor/sub/**" in out
    assert "third_party/thing/**" in out


def test_parse_gitmodules_returns_empty_when_missing(tmp_path: Path) -> None:
    """No ``.gitmodules`` → empty list (don't crash)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _parse_gitmodules(repo) == []


def test_parse_gitmodules_handles_malformed_file(tmp_path: Path) -> None:
    """Garbage in ``.gitmodules`` must not crash the walker — log + skip."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitmodules").write_text("this is not valid ini format @@@")
    assert _parse_gitmodules(repo) == []


def test_walk_repo_skips_gitmodule_paths(tmp_path: Path) -> None:
    """End-to-end: files inside a declared submodule path must not be yielded."""
    files = {
        "src/main.py": "def main(): pass\n",
        "vendor/sub/big.py": "def big(): pass\n",
        "vendor/sub/nested/deep.py": "def deep(): pass\n",
    }
    repo = _bootstrap_repo(tmp_path, files)
    (repo / ".gitmodules").write_text(
        '[submodule "vendor-sub"]\n'
        "    path = vendor/sub\n"
        "    url = https://example.invalid/sub.git\n"
    )
    cfg = load_config(repo)
    discovered = {str(p.relative_to(repo)) for p in discover_files(cfg)}
    assert "src/main.py" in discovered
    assert "vendor/sub/big.py" not in discovered, (
        f"submodule path leaked into discovery: {discovered}"
    )
    assert "vendor/sub/nested/deep.py" not in discovered
