"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from code_intel import __version__
from code_intel.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["init", "index", "serve", "doctor", "mcp-config", "install-services"]:
        assert cmd in result.output


def test_doctor_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--target" in result.output


def test_init_creates_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--target", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / ".codeindex" / "config.toml").exists()
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8").count(".codeindex/") == 1


def test_mcp_config_prints_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["mcp-config", "--target", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "mcpServers" in result.output
    assert "code-intel" in result.output
