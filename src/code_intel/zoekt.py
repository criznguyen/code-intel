"""Zoekt integration — STUB for v0.1.

In v0.1, ripgrep handles lexical search. Zoekt support is planned for v0.2 (see
README "Roadmap"). The interface below is intentionally documented so callers
can rely on a stable shape later.

Planned behaviour:
- `index_repo(cfg)` -> launch (or sync) a `sourcegraph/zoekt-indexserver` Docker
  container, point it at `cfg.target`, and store the trigram index under
  `cfg.zoekt.index_dir`.
- `search(cfg, query, max_results)` -> exec `zoekt` (or hit the HTTP gateway) and
  return [{path, line, snippet}] mirroring ripgrep results.
"""

from __future__ import annotations

from typing import Any

from code_intel.config import Config


def is_enabled(cfg: Config) -> bool:
    return cfg.zoekt.enabled


def index_repo(cfg: Config) -> None:  # pragma: no cover - stub
    raise NotImplementedError(
        "Zoekt integration is stubbed for v0.1 — use ripgrep via search_lexical. "
        "Planned in v0.2; see README roadmap."
    )


def search(
    cfg: Config, query: str, max_results: int = 50
) -> list[dict[str, Any]]:  # pragma: no cover - stub
    raise NotImplementedError("Zoekt search is stubbed for v0.1 — use search_lexical (ripgrep).")
