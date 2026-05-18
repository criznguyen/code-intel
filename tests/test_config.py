"""Roundtrip + defaults for config.toml."""

from __future__ import annotations

from pathlib import Path

from code_intel.config import (
    CODEINDEX_DIRNAME,
    CONFIG_FILENAME,
    Config,
    default_config,
    load_config,
    save_config,
)


def test_defaults_have_expected_keys() -> None:
    cfg = default_config(project_name="demo")
    assert cfg.project.name == "demo"
    assert cfg.embedding.provider == "ollama"
    assert cfg.embedding.model == "embeddinggemma"
    assert cfg.embedding.dim == 768
    assert ".codeindex/lancedb" in cfg.lancedb.path
    assert cfg.mcp.transport == "stdio"
    assert cfg.zoekt.enabled is False


def test_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    cfg = default_config(project_name="repo")
    cfg._target = target
    cfg.embedding.model = "nomic-embed-text"
    cfg.index.max_file_bytes = 500_000
    save_config(cfg, target=target)

    cfg_path = target / CODEINDEX_DIRNAME / CONFIG_FILENAME
    assert cfg_path.exists()

    reloaded = load_config(target)
    assert isinstance(reloaded, Config)
    assert reloaded.project.name == "repo"
    assert reloaded.embedding.model == "nomic-embed-text"
    assert reloaded.index.max_file_bytes == 500_000
    assert reloaded.target == target.resolve()


def test_load_without_config_returns_defaults(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    target.mkdir()
    cfg = load_config(target)
    assert cfg.project.name == "fresh"
    assert cfg.target == target.resolve()
    assert not cfg.config_path.exists()
