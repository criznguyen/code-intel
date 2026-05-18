"""Orchestrates: discover files -> chunk -> embed -> upsert."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pathspec

from code_intel._logging import get_logger
from code_intel.chunker import Chunk, chunk_file
from code_intel.config import Config
from code_intel.embedder import EmbedResult, get_provider

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


def _write_skipped_log(cfg: Config, result: EmbedResult, chunks: list[Chunk]) -> Path | None:
    """Persist per-chunk skip metadata to `.codeindex/skipped.jsonl`.

    Returns the path written, or None when no skips occurred.
    """
    if not result.skipped_indices:
        return None
    log_path = cfg.codeindex_dir / "skipped.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for idx in result.skipped_indices:
            c = chunks[idx]
            reason = result.skipped_reasons.get(idx, "unknown")
            f.write(
                json.dumps(
                    {
                        "path": c.path,
                        "symbol": c.symbol,
                        "lang": c.lang,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "chars": len(c.content),
                        "reason": reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return log_path


def index_repo(cfg: Config, since: str | None = None) -> dict[str, int]:
    """Run the full chunk-and-embed pipeline.

    Returns a stats dict: {files, chunks, embedded, skipped}.
    """
    files = discover_files(cfg, since=since)
    log.info("discovered %d files", len(files))

    all_chunks: list[Chunk] = []
    max_chunk_chars = cfg.index.max_chunk_chars
    for fp in files:
        chunks = chunk_file(
            fp, cfg.target, cfg.index.max_file_bytes, max_chunk_chars=max_chunk_chars
        )
        if chunks:
            all_chunks.extend(chunks)
    log.info("produced %d chunks", len(all_chunks))

    if not all_chunks:
        return {"files": len(files), "chunks": 0, "embedded": 0, "skipped": 0}

    provider = get_provider(cfg)
    texts = [c.content for c in all_chunks]
    result = provider.embed(texts)
    log.info(
        "embedded %d vectors via %s (skipped %d)",
        len(result.vectors),
        provider.name,
        len(result.skipped_indices),
    )

    # Filter out skipped chunks before upsert; their indices map 1:1 with `texts`.
    kept_chunks = [c for i, c in enumerate(all_chunks) if i not in result.skipped_indices]
    if len(kept_chunks) != len(result.vectors):
        # Defensive: provider contract bug. Refuse to corrupt the table.
        raise RuntimeError(
            f"embedder returned {len(result.vectors)} vectors but "
            f"{len(kept_chunks)} chunks survived skip-filtering"
        )

    _write_skipped_log(cfg, result, all_chunks)

    from code_intel.store import upsert_chunks

    written = upsert_chunks(cfg, kept_chunks, result.vectors)
    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "embedded": written,
        "skipped": len(result.skipped_indices),
    }
