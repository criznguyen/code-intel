"""LSP integration — STUB for v0.1.

Goal (v0.2+): spawn a `basedpyright`/`rust-analyzer`/`gopls` LSP client and expose
`go_to_definition`, `find_references`, and `hover` as MCP tools so the agent gets
ground-truth symbol resolution alongside the embedding-based search.

For v0.1, intentionally not implemented to keep the install footprint small and
the dependency surface minimal.
"""

from __future__ import annotations

from typing import Any

from code_intel.config import Config


def goto_definition(
    cfg: Config, file: str, line: int, character: int
) -> list[dict[str, Any]]:  # pragma: no cover
    raise NotImplementedError(
        "LSP integration is stubbed for v0.1 — use semantic_search + search_lexical instead."
    )


def find_references(
    cfg: Config, file: str, line: int, character: int
) -> list[dict[str, Any]]:  # pragma: no cover
    raise NotImplementedError("LSP integration is stubbed for v0.1 — see README roadmap.")


def hover(cfg: Config, file: str, line: int, character: int) -> str:  # pragma: no cover
    raise NotImplementedError("LSP integration is stubbed for v0.1 — see README roadmap.")
