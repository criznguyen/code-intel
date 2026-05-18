"""Orchestrates: discover files -> chunk -> embed -> upsert."""

from __future__ import annotations

import json
import os
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
    """Build a gitignore-style PathSpec from globs (supports `**/`).

    v0.1.6: migrated from deprecated ``gitwildmatch`` factory to
    :class:`pathspec.GitIgnoreSpec`. Behavior is identical for our glob
    patterns (no negation, no anchor-tail tricks) — only the implementation
    class changed.
    """
    return pathspec.GitIgnoreSpec.from_lines(globs)


def _matches_any(rel_path: str, spec: pathspec.PathSpec) -> bool:
    return spec.match_file(rel_path)


def _walk_repo(cfg: Config) -> Iterator[Path]:
    """Yield repo files honoring include/exclude globs with early dir pruning.

    v0.1.3 used ``rglob('*')`` which descends into every directory regardless
    of exclude patterns — recursing through ``target/`` or ``node_modules/``
    burns ~5-10 sec on large repos. We use ``os.walk`` with in-place
    ``dirs[:]`` pruning so excluded directories are never opened.
    (INFO-11 part A in v0.1.3 audit.)
    """
    root = cfg.target
    include_spec = _spec(cfg.index.include_globs)
    exclude_spec = _spec(cfg.index.exclude_globs)
    root_str = str(root)
    for dirpath, dirs, files in os.walk(root_str):
        dir_path = Path(dirpath)
        # Prune sub-dirs in place. gitwildmatch semantics need a trailing slash
        # plus relative-to-root form to anchor `**/target/**` correctly.
        kept_dirs: list[str] = []
        for d in dirs:
            child = dir_path / d
            try:
                rel = str(child.relative_to(root))
            except ValueError:
                kept_dirs.append(d)
                continue
            # Match dir path with trailing slash so `**/target/**` hits.
            if exclude_spec.match_file(rel + "/"):
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs
        for fname in files:
            p = dir_path / fname
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


def prune_orphans(cfg: Config) -> int:
    """Remove DB rows for paths that no longer match include/exclude globs
    (i.e. file deleted, renamed out, or globs tightened).

    Returns the number of rows removed. (MED-6 in v0.1.3 audit.)
    """
    from code_intel.store import delete_for_path, list_indexed_paths

    on_disk = {str(p.relative_to(cfg.target)) for p in _walk_repo(cfg)}
    indexed = list_indexed_paths(cfg)
    orphans = indexed - on_disk
    removed = 0
    for rel in orphans:
        removed += delete_for_path(cfg, rel)
    if orphans:
        log.info("pruned %d orphan path(s) from index", len(orphans))
    return removed


def index_repo(
    cfg: Config,
    since: str | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Run the full chunk-and-embed pipeline.

    When ``since`` is set, we content-hash-dedup against the existing table:
    for each chunk whose ``(path, symbol, start_line, content_hash)`` matches
    a stored row, we skip the embed call entirely (the row stays in place).
    Only changed or new chunks pay the embed cost. (INFO-11 part B, v0.1.5.)

    When ``force=True``, the dedup cache is bypassed (use after changing the
    embedding model / dim, where stored vectors are stale).

    Returns: {files, chunks, embedded, skipped, cache_hits}.
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
        return {
            "files": len(files),
            "chunks": 0,
            "embedded": 0,
            "skipped": 0,
            "cache_hits": 0,
        }

    # Content-hash dedup for incremental (--since) re-indexes only. A full
    # re-index always re-embeds — typical use is "embedding model changed".
    cache_hits = 0
    chunks_to_embed: list[Chunk] = all_chunks
    if since and not force:
        from code_intel.store import lookup_existing_hashes

        affected_paths = {c.path for c in all_chunks}
        existing = lookup_existing_hashes(cfg, affected_paths)
        misses: list[Chunk] = []
        for c in all_chunks:
            key = (c.path, c.symbol, int(c.start_line))
            if existing.get(key) == c.content_hash:
                cache_hits += 1
                continue
            misses.append(c)
        chunks_to_embed = misses
        log.info(
            "content-hash cache: %d/%d hits, %d to embed",
            cache_hits,
            len(all_chunks),
            len(chunks_to_embed),
        )

    if not chunks_to_embed:
        # Everything hit the cache: nothing to embed and nothing to upsert.
        # The existing rows stay in place; no delete-before-add for the
        # affected paths so cached rows survive.
        return {
            "files": len(files),
            "chunks": len(all_chunks),
            "embedded": 0,
            "skipped": 0,
            "cache_hits": cache_hits,
        }

    provider = get_provider(cfg)
    texts = [c.content for c in chunks_to_embed]
    result = provider.embed(texts)
    log.info(
        "embedded %d vectors via %s (skipped %d)",
        len(result.vectors),
        provider.name,
        len(result.skipped_indices),
    )

    # Filter out skipped chunks before upsert; their indices map 1:1 with `texts`.
    kept_chunks = [
        c for i, c in enumerate(chunks_to_embed) if i not in result.skipped_indices
    ]
    if len(kept_chunks) != len(result.vectors):
        # Defensive: provider contract bug. Refuse to corrupt the table.
        raise RuntimeError(
            f"embedder returned {len(result.vectors)} vectors but "
            f"{len(kept_chunks)} chunks survived skip-filtering"
        )

    _write_skipped_log(cfg, result, chunks_to_embed)

    # When we used the content-hash cache we MUST NOT delete-before-add the
    # affected paths (upsert_chunks does that per-path) — that would wipe out
    # the cached survivors. Instead, do a path-narrow delete only for paths
    # that have at least one miss, then add only the embedded misses.
    if since and not force and cache_hits > 0:
        from code_intel.store import _records, _open_or_create_table, open_db, _sql_quote

        if kept_chunks:
            rows = _records(kept_chunks, result.vectors)
            db = open_db(cfg)
            tbl = _open_or_create_table(db, cfg, sample_rows=rows)
            # Per-(path, symbol, start_line) selective delete so cache-hit
            # rows for the same file survive.
            ids_to_replace = {r["id"] for r in rows}
            if ids_to_replace:
                id_list = ",".join(_sql_quote(i) for i in ids_to_replace)
                try:
                    tbl.delete(f"id IN ({id_list})")
                except Exception as e:  # pragma: no cover
                    log.debug("selective delete before incremental add failed: %s", e)
            tbl.add(rows)
            written = len(rows)
        else:
            written = 0
    else:
        from code_intel.store import upsert_chunks

        written = upsert_chunks(cfg, kept_chunks, result.vectors)
    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "embedded": written,
        "skipped": len(result.skipped_indices),
        "cache_hits": cache_hits,
    }
