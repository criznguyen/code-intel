"""Orchestrates: discover files -> chunk -> embed -> upsert."""

from __future__ import annotations

import configparser
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


def _parse_gitmodules(root: Path) -> list[str]:
    """Return glob patterns for submodule paths declared in ``.gitmodules``.

    v0.1.8 Gap-3: vendored multi-MB submodules were walked transparently and
    embedded into the index, burning embed budget on third-party code the
    user typically does not want searchable. We parse ``.gitmodules`` (the
    canonical submodule manifest) and append ``<path>/**`` to runtime
    exclude globs so submodules are silently skipped.

    Returns ``[]`` when ``.gitmodules`` is missing or unparseable — we
    err on the side of "index everything" rather than swallow user data
    because of a malformed config file.
    """
    gm = root / ".gitmodules"
    if not gm.exists():
        return []
    cp = configparser.ConfigParser()
    try:
        cp.read(gm, encoding="utf-8")
    except (configparser.Error, OSError) as e:
        log.warning(".gitmodules unparseable, skipping submodule exclude: %s", e)
        return []
    paths: list[str] = []
    for sec in cp.sections():
        # Section headers look like: [submodule "name"]
        if not sec.startswith("submodule "):
            continue
        p = cp.get(sec, "path", fallback=None)
        if not p:
            continue
        # Normalize: strip leading/trailing slash, drop empty.
        p = p.strip().strip("/")
        if not p:
            continue
        paths.append(f"{p}/**")
    if paths:
        log.info("auto-excluding %d submodule path(s) from .gitmodules", len(paths))
    return paths


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
    # v0.1.8 Gap-3: append submodule paths from .gitmodules so vendored
    # third-party trees don't silently consume embed budget.
    exclude_spec = _spec(cfg.index.exclude_globs + _parse_gitmodules(root))
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
        # Same submodule-aware exclude treatment as _walk_repo.
        exclude_spec = _spec(cfg.index.exclude_globs + _parse_gitmodules(cfg.target))
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
    # v0.1.8 Gap-5: Fail fast on read-only / disk-full target BEFORE we burn
    # a full embed pass (can be ~30s+ on 2k chunks) only to crash on write.
    # `_check_lancedb_writable` returns `CheckResult(level=PASS|WARN|FAIL,
    # detail=...)`. We treat anything non-PASS as fatal.
    from code_intel.doctor import _check_lancedb_writable

    writable = _check_lancedb_writable(cfg)
    if writable.level != "PASS":
        raise RuntimeError(f"lancedb not writable: {writable.detail}")

    files = discover_files(cfg, since=since)
    log.info("discovered %d files", len(files))

    all_chunks: list[Chunk] = []
    chunker_skipped_files = 0  # v0.1.7: count files dropped by chunk_file (size cap)
    max_chunk_chars = cfg.index.max_chunk_chars
    max_file_bytes = cfg.index.max_file_bytes
    for fp in files:
        # Count oversize-skips up front so the final stats line is accurate
        # even when the produced-chunks fallthrough below short-circuits early.
        # detect_lang() returning None also makes chunk_file return [], but
        # those are filtered by include_globs already; size-cap is the one
        # silent skip we still need to surface.
        try:
            if fp.stat().st_size > max_file_bytes:
                chunker_skipped_files += 1
        except OSError:
            pass
        chunks = chunk_file(
            fp, cfg.target, max_file_bytes, max_chunk_chars=max_chunk_chars
        )
        if chunks:
            all_chunks.extend(chunks)
    log.info("produced %d chunks", len(all_chunks))
    if chunker_skipped_files:
        log.info(
            "chunker skipped %d file(s) over max_file_bytes=%d",
            chunker_skipped_files,
            max_file_bytes,
        )

    if not all_chunks:
        return {
            "files": len(files),
            "chunks": 0,
            "embedded": 0,
            "skipped": 0,
            "cache_hits": 0,
            "chunker_skipped_files": chunker_skipped_files,
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
            "chunker_skipped_files": chunker_skipped_files,
        }

    provider = get_provider(cfg)
    # v0.1.8 Gap-1: SIGINT batch-level checkpoint. Previously we called
    # ``provider.embed(all_texts)`` once and only upserted after the whole
    # corpus was embedded — a SIGINT mid-stream wasted *all* embed work.
    # We now split into checkpoint batches and commit each batch's surviving
    # rows immediately so a kill at batch N preserves batches 0..N-1.
    #
    # Sizing: pick ``max(provider.batch_size, 32)`` so each checkpoint
    # contains exactly one provider-internal Ollama batch — no double
    # batching, no smaller-than-Ollama-batch under-utilization.
    provider_batch_size = max(1, getattr(provider, "batch_size", 32))
    checkpoint_size = max(provider_batch_size, 32)

    # Aggregated result so the final stats / skip log match the v0.1.7
    # contract (single ``skipped.jsonl`` written at end, single stats dict).
    agg_result = EmbedResult()
    written = 0

    # Pre-resolve store imports once (selective-delete branch needs the
    # private helpers; the simple branch needs upsert_chunks). The selective
    # branch is taken only on incremental re-index with cache hits.
    use_selective = since and not force and cache_hits > 0
    if use_selective:
        from code_intel.store import (
            _open_or_create_table,
            _records,
            _sanitize_lance_error,
            _sql_quote,
            open_db,
        )
    else:
        from code_intel.store import upsert_chunks

    total_to_embed = len(chunks_to_embed)
    n_ckpt = (total_to_embed + checkpoint_size - 1) // checkpoint_size
    for ckpt_start in range(0, total_to_embed, checkpoint_size):
        batch_chunks = chunks_to_embed[ckpt_start : ckpt_start + checkpoint_size]
        batch_texts = [c.content for c in batch_chunks]
        batch_result = provider.embed(batch_texts)

        # Map this batch's skipped indices back to global indices so the
        # final skipped.jsonl is correct.
        for local_idx in batch_result.skipped_indices:
            global_idx = ckpt_start + local_idx
            agg_result.skipped_indices.append(global_idx)
            agg_result.skipped_reasons[global_idx] = batch_result.skipped_reasons.get(
                local_idx, "unknown"
            )
        agg_result.vectors.extend(batch_result.vectors)

        kept_batch = [
            c for i, c in enumerate(batch_chunks) if i not in batch_result.skipped_indices
        ]
        if len(kept_batch) != len(batch_result.vectors):
            raise RuntimeError(
                f"embedder returned {len(batch_result.vectors)} vectors but "
                f"{len(kept_batch)} chunks survived skip-filtering"
            )

        if not kept_batch:
            log.info(
                "checkpoint %d/%d: %d/%d embedded+committed (batch all skipped)",
                ckpt_start // checkpoint_size + 1,
                n_ckpt,
                written,
                total_to_embed,
            )
            continue

        if use_selective:
            rows = _records(kept_batch, batch_result.vectors)
            db = open_db(cfg)
            tbl = _open_or_create_table(db, cfg, sample_rows=rows)
            ids_to_replace = {r["id"] for r in rows}
            if ids_to_replace:
                id_list = ",".join(_sql_quote(i) for i in ids_to_replace)
                try:
                    tbl.delete(f"id IN ({id_list})")
                except Exception as e:  # pragma: no cover
                    log.debug(
                        "selective delete before incremental add failed: %s", e
                    )
            try:
                tbl.add(rows)
            except OSError as e:
                # Same sanitization as upsert_chunks; the selective branch
                # bypasses that wrapper so we duplicate the guard here.
                raise RuntimeError(
                    f"failed to upsert chunks: {_sanitize_lance_error(str(e))}"
                ) from e
            written += len(rows)
        else:
            written += upsert_chunks(cfg, kept_batch, batch_result.vectors)

        log.info(
            "checkpoint %d/%d: %d/%d embedded+committed",
            ckpt_start // checkpoint_size + 1,
            n_ckpt,
            written,
            total_to_embed,
        )

    log.info(
        "embedded %d vectors via %s (skipped %d)",
        len(agg_result.vectors),
        provider.name,
        len(agg_result.skipped_indices),
    )

    _write_skipped_log(cfg, agg_result, chunks_to_embed)

    return {
        "files": len(files),
        "chunks": len(all_chunks),
        "embedded": written,
        "skipped": len(agg_result.skipped_indices),
        "cache_hits": cache_hits,
        "chunker_skipped_files": chunker_skipped_files,
    }
