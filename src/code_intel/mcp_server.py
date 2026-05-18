"""FastMCP server exposing code-intel tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec

from code_intel._logging import get_logger
from code_intel.config import Config, load_config

log = get_logger(__name__)

_DIGEST_DIR = "digests"


def build_server(target: Path):
    """Build a FastMCP server bound to `target` repo. Returns the server instance."""
    from mcp.server.fastmcp import FastMCP

    cfg: Config = load_config(target)
    server = FastMCP(name=f"code-intel-{cfg.project.name}")

    @server.tool(description="Lexical (ripgrep) search across the repo.")
    def search_lexical(
        pattern: str,
        path_glob: str | None = None,
        lang: str | None = None,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        from code_intel.search import search_lexical as _do

        return _do(cfg, pattern, path_glob=path_glob, lang=lang, max_results=max_results)

    @server.tool(description="Semantic (vector) search using the indexed embeddings.")
    def semantic_search(query: str, k: int = 10, lang: str | None = None) -> list[dict[str, Any]]:
        from code_intel.search import semantic_search as _do

        return _do(cfg, query, k=k, lang=lang)

    @server.tool(description="Structural (ast-grep) search. Requires `lang` like 'rust'.")
    def structural(pattern: str, lang: str, max_results: int = 50) -> list[dict[str, Any]]:
        from code_intel.search import structural_search as _do

        return _do(cfg, pattern, lang=lang, max_results=max_results)

    @server.tool(description="Read a curated module digest from .codeindex/digests/.")
    def get_digest(module: str) -> str:
        digest = cfg.codeindex_dir / _DIGEST_DIR / f"{module}.md"
        if not digest.exists():
            return f"(no digest at {digest.relative_to(cfg.target)})"
        return digest.read_text(encoding="utf-8", errors="replace")

    @server.tool(description="List top-level modules (directories) under the repo root.")
    def list_modules() -> list[str]:
        root = cfg.target
        exclude_spec = pathspec.GitIgnoreSpec.from_lines(cfg.index.exclude_globs)
        mods: list[str] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if exclude_spec.match_file(entry.name + "/"):
                continue
            if entry.name.startswith("."):
                continue
            mods.append(entry.name)
        return mods

    return server


def run_stdio(target: Path) -> None:
    server = build_server(target)
    server.run(transport="stdio")
