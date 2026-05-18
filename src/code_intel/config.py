"""Pydantic config models + load/save helpers for code-intel."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel, Field

CONFIG_FILENAME = "config.toml"
CODEINDEX_DIRNAME = ".codeindex"


class ProjectSection(BaseModel):
    name: str = "project"
    root: str = "."


class IndexSection(BaseModel):
    include_globs: list[str] = Field(
        default_factory=lambda: [
            "**/*.rs",
            "**/*.py",
            "**/*.go",
            "**/*.ts",
            "**/*.tsx",
            "**/*.js",
            "**/*.jsx",
            "**/*.java",
            "**/*.kt",
            "**/*.md",
        ]
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            "**/target/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.codeindex/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
        ]
    )
    max_file_bytes: int = 1_000_000


class EmbeddingSection(BaseModel):
    # Core ships only "ollama"; external plugin packages may register their own
    # providers by implementing the EmbeddingProvider Protocol. Narrowing to a
    # Literal here makes config validation reject unknown providers at load time.
    provider: Literal["ollama"] = "ollama"
    model: str = "embeddinggemma"
    endpoint: str = "http://localhost:11434"
    batch_size: int = 32
    dim: int = 768


class LanceDBSection(BaseModel):
    path: str = ".codeindex/lancedb"
    table: str = "chunks"


class MCPSection(BaseModel):
    transport: str = "stdio"  # "stdio" | "unix-socket"
    socket_path: str = "/run/user/{uid}/code-intel-{project}.sock"


class ZoektSection(BaseModel):
    enabled: bool = False
    docker_image: str = "sourcegraph/zoekt-indexserver:latest"
    index_dir: str = ".codeindex/zoekt"


class Config(BaseModel):
    project: ProjectSection = Field(default_factory=ProjectSection)
    index: IndexSection = Field(default_factory=IndexSection)
    embedding: EmbeddingSection = Field(default_factory=EmbeddingSection)
    lancedb: LanceDBSection = Field(default_factory=LanceDBSection)
    mcp: MCPSection = Field(default_factory=MCPSection)
    zoekt: ZoektSection = Field(default_factory=ZoektSection)

    # --- resolved paths (not serialized) -----------------------------------
    _target: Path | None = None

    @property
    def target(self) -> Path:
        if self._target is None:
            raise RuntimeError("Config target not set — load via load_config(target).")
        return self._target

    @property
    def codeindex_dir(self) -> Path:
        return self.target / CODEINDEX_DIRNAME

    @property
    def lancedb_path(self) -> Path:
        p = Path(self.lancedb.path)
        return p if p.is_absolute() else self.target / p

    @property
    def config_path(self) -> Path:
        return self.codeindex_dir / CONFIG_FILENAME


def default_config(project_name: str | None = None) -> Config:
    cfg = Config()
    if project_name:
        cfg.project.name = project_name
    return cfg


def load_config(target: Path | str) -> Config:
    """Load config from `<target>/.codeindex/config.toml`. Falls back to defaults."""
    target_path = Path(target).resolve()
    cfg_path = target_path / CODEINDEX_DIRNAME / CONFIG_FILENAME
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        cfg = Config.model_validate(data)
    else:
        cfg = default_config(project_name=target_path.name)
    cfg._target = target_path
    return cfg


def save_config(cfg: Config, target: Path | str | None = None) -> Path:
    """Write config to `<target>/.codeindex/config.toml`. Returns the path."""
    target_path = Path(target).resolve() if target is not None else cfg.target
    cfg_dir = target_path / CODEINDEX_DIRNAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / CONFIG_FILENAME
    data = cfg.model_dump(mode="json")
    with cfg_path.open("wb") as f:
        tomli_w.dump(data, f)
    cfg._target = target_path
    return cfg_path
