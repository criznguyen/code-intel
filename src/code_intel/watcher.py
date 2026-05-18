"""File-watcher daemon. Incrementally re-indexes changed files."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pathspec

from code_intel._logging import get_logger
from code_intel.chunker import chunk_file
from code_intel.config import Config, load_config
from code_intel.embedder import get_provider

log = get_logger(__name__)

DEBOUNCE_SECONDS = 2.0


def _spec(globs: list[str]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitwildmatch", globs)


async def watch(target: Path) -> None:
    """Async loop: on file change, re-chunk+re-embed only changed files."""
    from watchfiles import awatch

    cfg: Config = load_config(target)
    root = cfg.target
    include_spec = _spec(cfg.index.include_globs)
    exclude_spec = _spec(cfg.index.exclude_globs)
    log.info("watching %s", root)

    async for changes in awatch(str(root), debounce=int(DEBOUNCE_SECONDS * 1000)):
        paths: set[Path] = set()
        for _ctype, raw_path in changes:
            p = Path(raw_path)
            if not p.exists() or not p.is_file():
                continue
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                continue
            if exclude_spec.match_file(rel):
                continue
            if not include_spec.match_file(rel):
                continue
            paths.add(p)
        if not paths:
            continue
        log.info("re-indexing %d changed file(s)", len(paths))
        await _reindex_files(cfg, list(paths))


async def _reindex_files(cfg: Config, files: list[Path]) -> None:
    chunks = []
    for f in files:
        chunks.extend(chunk_file(f, cfg.target, cfg.index.max_file_bytes))
    if not chunks:
        return
    provider = get_provider(cfg)
    texts = [c.content for c in chunks]
    # Embedder is sync; run in a thread to avoid blocking the loop.
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, provider.embed, texts)

    from code_intel.store import upsert_chunks

    written = await loop.run_in_executor(None, upsert_chunks, cfg, chunks, vectors)
    log.info("upserted %d chunks", written)


def run(target: Path) -> None:
    """Blocking entry point used by systemd / CLI."""
    asyncio.run(watch(target))
