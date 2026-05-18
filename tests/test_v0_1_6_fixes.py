"""Regression tests for the 6 items shipped in code-intel v0.1.6.

Items:
* NEW-1 — CLI ``index`` auto-elevates logging to INFO; embedder emits a
  per-batch progress log line for multi-batch embed calls.
* NEW-2 — INFO-10 RSS scale bench: documented only; see CHANGELOG for the
  measured plateau number.
* BL-1 — ``store._list_table_names`` normalizes LanceDB ``list_tables()``
  response across SDK versions; 0 ``table_names()`` deprecation warnings.
* BL-2 — ``pathspec.GitIgnoreSpec`` replaces deprecated ``gitwildmatch``
  factory in indexer / watcher / mcp_server; 0 ``GitWildMatchPattern``
  deprecation warnings.
* BL-3 — Solanabot reindex; documented only (operational task).
* BL-4 — ``code-intel search ... --no-rerank`` flag wired through to
  ``semantic_search(rerank=False)``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from unittest.mock import patch

import lancedb
import pathspec
import pytest
from typer.testing import CliRunner

from code_intel.cli import app
from code_intel.config import default_config
from code_intel.embedder import OllamaProvider
from code_intel.search import _reset_provider_cache, quick_cli_search
from code_intel.store import _list_table_names, _reset_db_cache

# ---------------------------------------------------------------------------
# BL-1 — list_tables() membership works across LanceDB versions
# ---------------------------------------------------------------------------


def test_list_table_names_returns_plain_list(tmp_path: Path) -> None:
    """`_list_table_names` must return a plain ``list[str]`` whether the
    underlying SDK gives a list (lancedb < 0.20) or a
    ``ListTablesResponse`` (lancedb >= 0.20)."""
    db_path = tmp_path / "lance"
    db_path.mkdir()
    db = lancedb.connect(str(db_path))
    assert _list_table_names(db) == []
    db.create_table("foo", data=[{"x": 1.0, "vector": [0.0] * 4}])
    names = _list_table_names(db)
    assert isinstance(names, list)
    assert "foo" in names


def test_no_table_names_deprecation_warning_on_open_table(tmp_path: Path) -> None:
    """Opening / re-opening a table must not surface
    ``DeprecationWarning: table_names() is deprecated``."""
    _reset_db_cache()
    cfg = default_config(project_name="bl1")
    cfg._target = tmp_path
    cfg.lancedb.path = str(tmp_path / "db")

    from code_intel.chunker import Chunk
    from code_intel.store import upsert_chunks

    chunk = Chunk(
        path="x.py",
        lang="python",
        symbol="f",
        kind="function",
        start_line=1,
        end_line=2,
        content="def f(): return 1\n",
        content_hash="abc",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        upsert_chunks(cfg, [chunk], [[0.1] * cfg.embedding.dim])
        upsert_chunks(cfg, [chunk], [[0.1] * cfg.embedding.dim])
    msgs = [str(w.message) for w in caught]
    assert not any("table_names() is deprecated" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# BL-2 — GitIgnoreSpec replaces gitwildmatch factory
# ---------------------------------------------------------------------------


def test_indexer_spec_uses_gitignorespec() -> None:
    """`indexer._spec` must build a GitIgnoreSpec, not the deprecated
    GitWildMatchPattern variant."""
    from code_intel.indexer import _spec

    spec = _spec(["**/*.py", "!tests/**"])
    assert isinstance(spec, pathspec.GitIgnoreSpec)


def test_watcher_spec_uses_gitignorespec() -> None:
    from code_intel.watcher import _spec

    spec = _spec(["**/*.rs"])
    assert isinstance(spec, pathspec.GitIgnoreSpec)


def test_no_gitwildmatch_deprecation_in_walk(tmp_path: Path) -> None:
    """A full walk of a small tree must not surface
    ``GitWildMatchPattern ('gitwildmatch') is deprecated``."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.js").write_text("// vendored\n")

    cfg = default_config(project_name="bl2")
    cfg._target = tmp_path
    cfg.index.include_globs = ["**/*.py", "**/*.js"]
    cfg.index.exclude_globs = ["**/node_modules/**"]

    from code_intel.indexer import discover_files

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        files = discover_files(cfg)
    msgs = [str(w.message) for w in caught]
    assert not any("gitwildmatch" in m.lower() for m in msgs), msgs
    rels = {p.relative_to(tmp_path).as_posix() for p in files}
    assert "src/a.py" in rels
    assert "node_modules/b.js" not in rels


# ---------------------------------------------------------------------------
# NEW-1 — CLI index auto-INFO + per-batch embedder log
# ---------------------------------------------------------------------------


def test_embedder_emits_per_batch_progress_log(caplog: pytest.LogCaptureFixture) -> None:
    """Multi-batch embed call must emit an INFO "ollama embed batch X/Y" line
    so a foreground ``--full`` reindex isn't visually frozen."""

    class _StubBatch(OllamaProvider):  # type: ignore[misc]
        def __init__(self):
            self.endpoint = "http://stub"
            self.model = "stub"
            self.batch_size = 2
            self.dim = 3
            self._batch_endpoint_disabled = False

        def _embed_batch(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    prov = _StubBatch()
    with caplog.at_level(logging.INFO, logger="code_intel.embedder"):
        res = prov.embed(["a", "b", "c", "d", "e"])
    assert len(res.vectors) == 5
    batch_logs = [r for r in caplog.records if "ollama embed batch" in r.getMessage()]
    # 5 items / batch_size 2 → 3 batches.
    assert len(batch_logs) >= 3, [r.getMessage() for r in caplog.records]


def test_embedder_skips_batch_log_for_single_query() -> None:
    """Single-batch embed (e.g. a query at search time) must NOT spam a
    progress line — only multi-batch indexer runs should."""

    class _StubBatch(OllamaProvider):  # type: ignore[misc]
        def __init__(self):
            self.endpoint = "http://stub"
            self.model = "stub"
            self.batch_size = 64
            self.dim = 3
            self._batch_endpoint_disabled = False

        def _embed_batch(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    prov = _StubBatch()
    log = logging.getLogger("code_intel.embedder")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.INFO)
    log.addHandler(handler)
    try:
        prov.embed(["only one query"])
    finally:
        log.removeHandler(handler)
    assert not any("ollama embed batch" in r.getMessage() for r in records)


def test_cli_index_auto_elevates_to_info(tmp_path: Path, monkeypatch) -> None:
    """CLI ``index`` must elevate root logger to INFO when no explicit
    CODE_INTEL_LOG override is set, so progress logs surface without
    needing ``-v``."""
    monkeypatch.delenv("CODE_INTEL_LOG", raising=False)
    # Force root logger above INFO so we can detect the auto-elevation.
    logging.getLogger().setLevel(logging.WARNING)

    cfg_dir = tmp_path / ".codeindex"
    cfg_dir.mkdir(parents=True)
    runner = CliRunner()
    # `init` first so config exists.
    init_res = runner.invoke(app, ["init", "--target", str(tmp_path), "--force"])
    assert init_res.exit_code == 0, init_res.output

    # Patch index_repo so the test doesn't need a real Ollama server.
    seen_level: dict[str, int] = {}

    def _fake_index_repo(cfg, since=None, force=False):
        seen_level["lvl"] = logging.getLogger().getEffectiveLevel()
        return {"files": 0, "chunks": 0, "embedded": 0, "skipped": 0, "cache_hits": 0}

    def _fake_prune(cfg):
        return 0

    with (
        patch("code_intel.indexer.index_repo", _fake_index_repo),
        patch("code_intel.indexer.prune_orphans", _fake_prune),
    ):
        res = runner.invoke(app, ["index", "--target", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert seen_level.get("lvl") == logging.INFO, seen_level


# ---------------------------------------------------------------------------
# BL-4 — CLI --rerank/--no-rerank flag
# ---------------------------------------------------------------------------


def test_cli_search_no_rerank_passes_through(tmp_path: Path) -> None:
    """`code-intel search --no-rerank QUERY` must reach
    `semantic_search(rerank=False)`."""
    runner = CliRunner()
    init_res = runner.invoke(app, ["init", "--target", str(tmp_path), "--force"])
    assert init_res.exit_code == 0, init_res.output

    captured: dict[str, object] = {}

    def _fake_semantic_search(cfg, query, k=10, lang=None, rerank=True):
        captured["query"] = query
        captured["k"] = k
        captured["rerank"] = rerank
        return [
            {
                "path": "a.py",
                "symbol": "f",
                "kind": "function",
                "lang": "python",
                "start_line": 1,
                "end_line": 2,
                "content": "def f(): ...",
                "score": 0.1,
                "rerank_score": None,
            }
        ]

    _reset_provider_cache()
    with patch("code_intel.search.semantic_search", _fake_semantic_search):
        res = runner.invoke(
            app,
            ["search", "calculate fee", "--target", str(tmp_path), "--no-rerank", "--k", "3"],
        )
    assert res.exit_code == 0, res.output
    assert captured["rerank"] is False
    assert captured["k"] == 3
    assert "rerank=off" in res.output


def test_cli_search_default_keeps_rerank_on(tmp_path: Path) -> None:
    """Default invocation must still pass ``rerank=True`` (backward-compat)."""
    runner = CliRunner()
    init_res = runner.invoke(app, ["init", "--target", str(tmp_path), "--force"])
    assert init_res.exit_code == 0, init_res.output

    captured: dict[str, object] = {}

    def _fake_semantic_search(cfg, query, k=10, lang=None, rerank=True):
        captured["rerank"] = rerank
        return []

    with patch("code_intel.search.semantic_search", _fake_semantic_search):
        res = runner.invoke(app, ["search", "x", "--target", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert captured["rerank"] is True


def test_quick_cli_search_rerank_param_forwards() -> None:
    """`quick_cli_search` must forward its ``rerank`` arg to
    ``semantic_search`` so the CLI flag is not silently dropped."""
    cfg = default_config(project_name="bl4")
    cfg._target = Path("/tmp/bl4-fake")

    seen: dict[str, object] = {}

    def _fake(cfg_, query, k=10, lang=None, rerank=True):
        seen["rerank"] = rerank
        return []

    with patch("code_intel.search.semantic_search", _fake):
        quick_cli_search(cfg, "q", k=2, rerank=False)
    assert seen["rerank"] is False
