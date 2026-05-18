"""Orchestrates: discover files -> chunk -> embed -> upsert."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pathspec

from code_intel._logging import get_logger
from code_intel.chunker import Chunk, chunk_file
from code_intel.config import Config
from code_intel.embedder import get_provider

log = get_logger(__name__)


def _spec(globs: list[str]) -> pathspec.PathSpec:
    """Build a gitwildmatch-style PathSpec from globs (supports `**/`)."""
    return pathspec.PathSpec.from_lines("gitwildmatch", globs)


def _matches_any(rel_path: str, spec: pathspec.PathSpec) -> bool:
    return spec.match_file(rel_path)


def _walk_repo(cfg: Config) -> Iterator[Path]:
    root = cfg.target
    include_spec = _spec(cfg.index.include_globs)
    exclude_spec = _spec(cfg.index.exclude_globs)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            continue
        if exclude_spec.match_file(rel):
            continue
        if not include_spec.match_file(rel):
            continue
        yield p


def _git_changed_files(cfg: Config, since: str) -> list[Path]:
    """Return files changed since `since` (git ref). Includes untracked."""
    root = cfg.target
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", f"{since}..HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        diff_uncommitted = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning("git diff failed: %s", e.stderr)
        return []
    names = set(
        filter(
            None,
            (diff.stdout + "\n" + diff_uncommitted.stdout + "\n" + untracked.stdout).splitlines(),
        )
    )
    return [root / n for n in names if (root / n).is_file()]


def discover_files(cfg: Config, since: str | None = None) -> list[Path]:
    if since:
        candidates = _git_changed_files(cfg, since)
        include_spec = _spec(cfg.index.include_globs)
        exclude_spec = _spec(cfg.index.exclude_globs)
        keep: list[Path] = []
        for p in candidates:
            try:
                rel = str(p.relative_to(cfg.target))
            except ValueError:
                continue
            if exclude_spec.match_file(rel):
                continue
            if not include_spec.match_file(rel):
                continue
            keep.append(p)
        return keep
    return list(_walk_repo(cfg))


def index_repo(cfg: Config, since: str | None = None) -> dict[str, int]:
    """Run the full chunk-and-embed pipeline.

    Returns a stats dict: {files, chunks, embedded}.
    """
    files = discover_files(cfg, since=since)
    log.info("discovered %d files", len(files))

    all_chunks: list[Chunk] = []
    for fp in files:
        chunks = chunk_file(fp, cfg.target, cfg.index.max_file_bytes)
        if chunks:
            all_chunks.extend(chunks)
    log.info("produced %d chunks", len(all_chunks))

    if not all_chunks:
        return {"files": len(files), "chunks": 0, "embedded": 0}

    provider = get_provider(cfg)
    texts = [c.content for c in all_chunks]
    vectors = provider.embed(texts)
    log.info("embedded %d vectors via %s", len(vectors), provider.name)

    from code_intel.store import upsert_chunks

    written = upsert_chunks(cfg, all_chunks, vectors)
    return {"files": len(files), "chunks": len(all_chunks), "embedded": written}
