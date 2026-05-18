"""Regression tests for the 3 items shipped in code-intel v0.1.7.

Items:
* NEW-1 (P1 perf) — ``MALLOC_ARENA_MAX=2`` re-exec bootstrap in ``cli.py``
  caps glibc per-thread arenas before any native lib alloc. Plateau RSS on
  the solanabot bench dropped from 589 MB → ~220 MB (62 % cut) with no
  measured latency regression. Tested via:
    1. ``_should_arena_bootstrap`` returns ``False`` when imported from
       ``python -c`` / interactive / tests (argv[0] not a real file).
    2. ``_should_arena_bootstrap`` returns ``True`` when argv[0] is a real
       file and the sentinel env var is unset.
    3. The opt-out env var ``CODE_INTEL_DISABLE_ARENA_CAP=1`` is honored.
    4. The sentinel ``_CODE_INTEL_ARENA_BOOTSTRAPPED=1`` prevents re-exec.
* NEW-2 (MED) — ``chunker.chunk_file`` emits a WARNING log when a file is
  dropped for exceeding ``max_file_bytes``. ``indexer.index_repo`` returns
  ``chunker_skipped_files`` count in its stats dict. CLI surfaces it in
  the final ``indexed`` line.
* NEW-3 (MED) — ``config.ZoektSection.enabled = true`` is rejected at
  load-time with a clear "reserved for v0.2" message. Default remains
  ``false``, matching the README roadmap commitment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from code_intel import cli as cli_module
from code_intel.chunker import chunk_file
from code_intel.config import ZoektSection, load_config
from code_intel.indexer import index_repo

# ---------------------------------------------------------------------------
# NEW-1 — MALLOC_ARENA_MAX bootstrap gating
# ---------------------------------------------------------------------------


def test_arena_bootstrap_skipped_when_argv0_is_dash_c(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the CLI module under ``python -c`` (argv[0] == '-c') MUST
    NOT trigger a re-exec — that would lose the -c payload entirely."""
    monkeypatch.setattr(sys, "argv", ["-c"])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    assert cli_module._should_arena_bootstrap() is False


def test_arena_bootstrap_skipped_when_argv0_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive / REPL embedded use → argv[0] is empty string."""
    monkeypatch.setattr(sys, "argv", [""])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    assert cli_module._should_arena_bootstrap() is False


def test_arena_bootstrap_skipped_when_sentinel_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After re-exec the sentinel env var is set; second-pass MUST short-circuit."""
    fake_script = tmp_path / "code-intel"
    fake_script.write_text("#!stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_script)])
    monkeypatch.setenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", "1")
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    assert cli_module._should_arena_bootstrap() is False


def test_arena_bootstrap_skipped_when_opt_out_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``CODE_INTEL_DISABLE_ARENA_CAP=1`` is the operator escape hatch."""
    fake_script = tmp_path / "code-intel"
    fake_script.write_text("#!stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_script)])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.setenv("CODE_INTEL_DISABLE_ARENA_CAP", "1")
    assert cli_module._should_arena_bootstrap() is False


def test_arena_bootstrap_fires_for_real_cli_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """argv[0] is the installed ``code-intel`` shim, sentinel unset → re-exec.

    Whitelist matches by basename: ``code-intel`` (installed shim),
    ``__main__.py`` (``python -m code_intel``), or ``bench_memory.py``.
    """
    fake_shim = tmp_path / "code-intel"
    fake_shim.write_text("#!stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_shim), "search", "foo"])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli_module._should_arena_bootstrap() is True


def test_arena_bootstrap_skipped_for_pytest_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even though pytest's binary IS a real file, we MUST NOT re-exec it —
    that would shred the in-flight test collector state. Whitelist gate."""
    fake_pytest = tmp_path / "pytest"
    fake_pytest.write_text("#!stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_pytest)])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli_module._should_arena_bootstrap() is False


def test_arena_bootstrap_fires_for_python_dash_m_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``python -m code_intel`` sets argv[0] to the __main__.py absolute path."""
    fake_main = tmp_path / "__main__.py"
    fake_main.write_text("# stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_main)])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert cli_module._should_arena_bootstrap() is True


def test_arena_bootstrap_skipped_on_non_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MALLOC_ARENA_MAX is glibc-specific; macOS / Windows must skip cleanly."""
    fake_script = tmp_path / "code-intel"
    fake_script.write_text("#!stub\n")
    monkeypatch.setattr(sys, "argv", [str(fake_script)])
    monkeypatch.delenv("_CODE_INTEL_ARENA_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("CODE_INTEL_DISABLE_ARENA_CAP", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cli_module._should_arena_bootstrap() is False


# ---------------------------------------------------------------------------
# NEW-2 — chunker over-max_file_bytes file emits warning + counter
# ---------------------------------------------------------------------------


def test_file_over_max_bytes_logs_warning_and_returns_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A file larger than max_bytes must emit a clear WARNING and return []."""
    big = tmp_path / "big.py"
    # Use a syntactically valid Python file so lang detection succeeds.
    big.write_text("x = 1\n" * 200_000)  # ~1.2 MB
    assert big.stat().st_size > 100_000

    caplog.set_level(logging.WARNING, logger="code_intel.chunker")
    out = chunk_file(big, tmp_path, max_bytes=100_000)
    assert out == []
    assert any(
        "chunker skip" in r.message and "max_file_bytes" in r.message for r in caplog.records
    ), f"expected chunker-skip warning, got: {[r.message for r in caplog.records]}"


def test_index_repo_reports_chunker_skipped_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``index_repo`` stats dict must include ``chunker_skipped_files`` so the
    CLI / API caller can surface it without parsing logs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    # Small file (kept) and big file (skipped).
    (repo / "small.py").write_text("def foo():\n    return 1\n")
    (repo / "big.py").write_text("x = 1\n" * 200_000)

    from code_intel.config import default_config, save_config

    cfg = default_config("test")
    cfg._target = repo
    cfg.index.max_file_bytes = 50_000  # forces big.py to be skipped
    cfg.index.include_globs = ["**/*.py"]
    save_config(cfg, target=repo)
    cfg = load_config(repo)
    cfg.index.max_file_bytes = 50_000

    # Stub embedder so we don't need Ollama for this test.
    from code_intel import indexer as indexer_mod

    class _StubProvider:
        name = "stub"
        dim = 768

        def embed(self, texts):
            from code_intel.embedder import EmbedResult

            return EmbedResult(vectors=[[0.0] * 768 for _ in texts])

    monkeypatch.setattr(indexer_mod, "get_provider", lambda _cfg: _StubProvider())

    stats = index_repo(cfg)
    assert "chunker_skipped_files" in stats
    assert stats["chunker_skipped_files"] == 1
    # small.py should have made it through.
    assert stats["files"] >= 1


# ---------------------------------------------------------------------------
# NEW-3 — ZoektSection model_validator rejects enabled=True
# ---------------------------------------------------------------------------


def test_zoekt_enabled_rejected_until_v0_2() -> None:
    """Loading ``zoekt.enabled = true`` from config must raise ValueError
    pointing at the v0.2 roadmap."""
    with pytest.raises(ValueError) as excinfo:
        ZoektSection(enabled=True)
    msg = str(excinfo.value)
    assert "v0.2" in msg, f"error message must reference v0.2 roadmap: {msg}"


def test_zoekt_disabled_default_still_loads() -> None:
    """``enabled = false`` (the default) must continue to load cleanly."""
    z = ZoektSection()
    assert z.enabled is False
    z2 = ZoektSection(enabled=False)
    assert z2.enabled is False


def test_zoekt_config_with_enabled_true_in_toml_rejected(tmp_path: Path) -> None:
    """End-to-end: ``load_config`` against a config.toml that sets
    ``zoekt.enabled = true`` must fail with the v0.2 message."""
    cfg_dir = tmp_path / ".codeindex"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text("[zoekt]\nenabled = true\n")
    with pytest.raises(ValueError) as excinfo:
        load_config(tmp_path)
    assert "v0.2" in str(excinfo.value)
