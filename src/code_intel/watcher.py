"""File-watcher daemon. Incrementally re-indexes changed files."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pathspec

from code_intel._logging import get_logger, setup_logging
from code_intel.chunker import chunk_file
from code_intel.config import Config, load_config
from code_intel.embedder import get_provider

log = get_logger(__name__)

DEBOUNCE_SECONDS = 2.0


def _spec(globs: list[str]) -> pathspec.PathSpec:
    # v0.1.6: GitIgnoreSpec replaces deprecated gitwildmatch factory.
    return pathspec.GitIgnoreSpec.from_lines(globs)


async def watch(target: Path) -> None:
    """Async loop: on file change, re-chunk+re-embed only changed files.

    Deletes are tracked separately and routed through
    :func:`code_intel.store.delete_for_path` so orphan chunks don't linger
    after a rename or rm. (MED-6 in v0.1.3 audit.)
    """
    from watchfiles import Change, awatch

    cfg: Config = load_config(target)
    root = cfg.target
    include_spec = _spec(cfg.index.include_globs)
    exclude_spec = _spec(cfg.index.exclude_globs)
    log.info("watching %s", root)

    async for changes in awatch(str(root), debounce=int(DEBOUNCE_SECONDS * 1000)):
        paths: set[Path] = set()
        deleted_rels: set[str] = set()
        for ctype, raw_path in changes:
            p = Path(raw_path)
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                continue
            if exclude_spec.match_file(rel):
                continue
            if not include_spec.match_file(rel):
                continue
            if ctype == Change.deleted:
                deleted_rels.add(rel)
                continue
            if not p.exists() or not p.is_file():
                # add/modify event but file already gone (race) — treat as delete.
                deleted_rels.add(rel)
                continue
            paths.add(p)
        if deleted_rels:
            log.info("pruning %d deleted file(s) from index", len(deleted_rels))
            from code_intel.store import delete_for_path

            loop = asyncio.get_running_loop()
            for rel in deleted_rels:
                await loop.run_in_executor(None, delete_for_path, cfg, rel)
        if not paths:
            continue
        log.info("re-indexing %d changed file(s)", len(paths))
        await _reindex_files(cfg, list(paths))


async def _reindex_files(cfg: Config, files: list[Path]) -> None:
    chunks = []
    max_chunk_chars = cfg.index.max_chunk_chars
    for f in files:
        chunks.extend(
            chunk_file(f, cfg.target, cfg.index.max_file_bytes, max_chunk_chars=max_chunk_chars)
        )
    if not chunks:
        return
    provider = get_provider(cfg)
    texts = [c.content for c in chunks]
    # Embedder is sync; run in a thread to avoid blocking the loop.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, provider.embed, texts)
    if result.skipped_indices:
        log.warning(
            "watcher: skipped %d chunk(s) during re-embed; survivors=%d",
            len(result.skipped_indices),
            len(result.vectors),
        )
    kept = [c for i, c in enumerate(chunks) if i not in result.skipped_indices]
    if not kept:
        return

    from code_intel.store import upsert_chunks

    written = await loop.run_in_executor(None, upsert_chunks, cfg, kept, result.vectors)
    log.info("upserted %d chunks", written)


def run(target: Path) -> None:
    """Blocking entry point used by systemd / CLI."""
    asyncio.run(watch(target))


def main() -> None:
    """Entrypoint for `python -m code_intel.watcher <target>`."""
    setup_logging("INFO")
    if len(sys.argv) < 2:
        sys.exit("usage: python -m code_intel.watcher <target_path>")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
