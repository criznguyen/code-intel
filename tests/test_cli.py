"""CLI smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from code_intel import __version__
from code_intel.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_help() -> None:
    import re

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    collapsed = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    for cmd in ["init", "index", "serve", "doctor", "mcp-config", "install-services"]:
        assert cmd in collapsed


def test_doctor_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    # Strip ANSI + collapse whitespace so we don't depend on TTY width wrapping.
    import re

    collapsed = re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert "--target" in collapsed
    assert "Health-check" in collapsed


def test_init_creates_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--target", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".codeindex" / "config.toml").exists()
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8").count(".codeindex/") == 1


def test_mcp_config_project_scope_uses_bare_key(tmp_path: Path) -> None:
    """Project-scope MCP entries already namespace by repo path → key='code-intel'."""
    result = runner.invoke(app, ["mcp-config", "--target", str(tmp_path), "--scope", "project"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    keys = list(payload["mcpServers"].keys())
    assert keys == ["code-intel"], keys


def test_mcp_config_user_scope_namespaces_by_project(tmp_path: Path) -> None:
    """User-scope MCP entries collide across projects, so we keep the name suffix."""
    project = tmp_path / "myrepo"
    project.mkdir()
    result = runner.invoke(app, ["mcp-config", "--target", str(project), "--scope", "user"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    keys = list(payload["mcpServers"].keys())
    assert keys == ["code-intel-myrepo"], keys


def test_install_services_force_writes_absolute_python_path(tmp_path: Path) -> None:
    """install-services --force must render ExecStart with an absolute interpreter,
    NOT a bare `python` token (regression guard for v0.1.x systemd bug)."""
    fake_unit_dir = tmp_path / "systemd_user"
    fake_projects_dir = tmp_path / "projects"
    target = tmp_path / "x"
    target.mkdir()

    with (
        patch("code_intel.systemd.USER_UNIT_DIR", fake_unit_dir),
        patch("code_intel.systemd.PROJECTS_DIR", fake_projects_dir),
    ):
        result = runner.invoke(
            app,
            [
                "install-services",
                "--force",
                "--instance",
                "test",
                "--target",
                str(target),
            ],
        )
    assert result.exit_code == 0, result.output

    watcher_unit = (fake_unit_dir / "code-intel-watcher@.service").read_text()
    mcp_unit = (fake_unit_dir / "code-intel-mcp@.service").read_text()

    # ExecStart line must contain an absolute path, never `exec python `
    # (bare token would resolve via systemd user-session PATH, which doesn't
    # have pyenv shims and silently exits 127).
    for line in watcher_unit.splitlines():
        if line.startswith("ExecStart="):
            assert " exec python " not in line, line
            assert "{{ python_bin }}" not in line, line
            assert "/" in line, line  # contains a path
    for line in mcp_unit.splitlines():
        if line.startswith("ExecStart="):
            assert " exec code-intel " not in line, line
            assert "{{ code_intel_bin }}" not in line, line

    # Project manifest must be written.
    assert (fake_projects_dir / "test.toml").exists()


def test_install_services_without_force_preserves_existing(tmp_path: Path) -> None:
    """Without --force, an existing unit file is NOT overwritten (safety guard)."""
    fake_unit_dir = tmp_path / "systemd_user"
    fake_unit_dir.mkdir()
    fake_projects_dir = tmp_path / "projects"
    target = tmp_path / "x"
    target.mkdir()

    sentinel = "EXISTING_USER_OVERRIDE\n"
    (fake_unit_dir / "code-intel-watcher@.service").write_text(sentinel)

    with (
        patch("code_intel.systemd.USER_UNIT_DIR", fake_unit_dir),
        patch("code_intel.systemd.PROJECTS_DIR", fake_projects_dir),
    ):
        result = runner.invoke(
            app,
            ["install-services", "--instance", "test", "--target", str(target)],
        )
    assert result.exit_code == 0
    assert (fake_unit_dir / "code-intel-watcher@.service").read_text() == sentinel


# ---------------------------------------------------------------------------
# Gap 8 (LOW): `code-intel doctor --strict` — treat WARN as FAIL for CI.
# Default behavior must remain unchanged (backward compat).
# ---------------------------------------------------------------------------
def _warn_only_results():
    from code_intel.doctor import CheckResult

    return [
        CheckResult("ripgrep", "PASS", "ripgrep 14.0.0"),
        CheckResult(
            "ollama-model",
            "WARN",
            "model 'embeddinggemma:latest' not pulled. Run `ollama pull embeddinggemma:latest`",
        ),
    ]


def test_doctor_warns_on_missing_model_default(tmp_path: Path) -> None:
    """Default (no --strict): WARN-only results must exit 0 + show the WARN line."""
    with patch("code_intel.doctor.run_doctor", return_value=_warn_only_results()):
        result = runner.invoke(app, ["doctor", "--target", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "WARN" in result.output
    assert "ollama-model" in result.output


def test_doctor_fails_on_missing_model_with_strict(tmp_path: Path) -> None:
    """--strict: a single WARN must escalate to exit 1 + print the strict banner."""
    with patch("code_intel.doctor.run_doctor", return_value=_warn_only_results()):
        result = runner.invoke(app, ["doctor", "--target", str(tmp_path), "--strict"])
    assert result.exit_code == 1, result.output
    assert "WARN" in result.output
    assert "--strict" in result.output


def test_has_warnings_helper() -> None:
    """has_warnings flips on at least one WARN; has_failures stays False on WARN-only."""
    from code_intel.doctor import CheckResult, has_failures, has_warnings

    pass_only = [CheckResult("a", "PASS", "ok")]
    warn_only = [CheckResult("a", "PASS", "ok"), CheckResult("b", "WARN", "missing")]
    fail_mixed = [CheckResult("a", "WARN", "missing"), CheckResult("b", "FAIL", "broken")]

    assert has_warnings(pass_only) is False
    assert has_failures(pass_only) is False

    assert has_warnings(warn_only) is True
    assert has_failures(warn_only) is False

    assert has_warnings(fail_mixed) is True
    assert has_failures(fail_mixed) is True
