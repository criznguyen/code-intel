"""LanceDB read/write helpers."""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from code_intel._logging import get_logger
from code_intel.chunker import Chunk
from code_intel.config import Config

log = get_logger(__name__)


def _sql_quote(s: str) -> str:
    """SQL-quote a string literal by doubling embedded apostrophes.

    Prevents path values like ``it's.rs`` from breaking LanceDB SQL filters
    (and from being silently treated as duplicate rows because the parse
    error swallowed the per-path delete-before-add step).
    """
    return "'" + s.replace("'", "''") + "'"


def chunk_id(chunk: Chunk) -> str:
    """Stable id for upsert: hash(path+symbol+start_line)."""
    key = f"{chunk.path}::{chunk.symbol}::{chunk.start_line}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _records(chunks: list[Chunk], vectors: list[list[float]]) -> list[dict[str, Any]]:
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) vs vectors ({len(vectors)}) length mismatch")
    now = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for c, v in zip(chunks, vectors, strict=True):
        rows.append(
            {
                "id": chunk_id(c),
                "path": c.path,
                "lang": c.lang,
                "symbol": c.symbol,
                "kind": c.kind,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "content_hash": c.content_hash,
                "vector": v,
                "indexed_at": now,
            }
        )
    return rows


_DB_CACHE: dict[tuple[str, str], Any] = {}
_DB_CACHE_LOCK = threading.Lock()


def open_db(cfg: Config):
    """Lazy import lancedb; open DB at configured path.

    Connections are cached per (resolved_path, table_name). LanceDB's
    connection object is process-local and re-opening it on every query
    burns ~10ms + ~1MB allocation per call (LOW-9 in v0.1.3 audit).
    """
    import lancedb  # type: ignore

    db_path = cfg.lancedb_path
    db_path.mkdir(parents=True, exist_ok=True)
    key = (str(db_path.resolve()), cfg.lancedb.table)
    with _DB_CACHE_LOCK:
        cached = _DB_CACHE.get(key)
        if cached is not None:
            return cached
        conn = lancedb.connect(str(db_path))
        _DB_CACHE[key] = conn
        return conn


def _reset_db_cache() -> None:
    """Test-only: drop cached connections so per-tmp_path tests stay isolated."""
    with _DB_CACHE_LOCK:
        _DB_CACHE.clear()


def _open_or_create_table(db, cfg: Config, sample_rows: list[dict[str, Any]] | None = None):
    name = cfg.lancedb.table
    if name in db.table_names():
        return db.open_table(name)
    if sample_rows:
        return db.create_table(name, data=sample_rows)
    # Empty table — write a single placeholder then delete it so schema sticks.
    zero = [0.0] * cfg.embedding.dim
    placeholder = {
        "id": "__init__",
        "path": "",
        "lang": "",
        "symbol": "",
        "kind": "",
        "start_line": 0,
        "end_line": 0,
        "content": "",
        "content_hash": "",
        "vector": zero,
        "indexed_at": datetime.now(UTC).isoformat(),
    }
    tbl = db.create_table(name, data=[placeholder])
    tbl.delete("id = '__init__'")
    return tbl


def upsert_chunks(cfg: Config, chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Upsert chunks (delete-then-add by path-set for v0.1 simplicity)."""
    if not chunks:
        return 0
    rows = _records(chunks, vectors)
    db = open_db(cfg)
    tbl = _open_or_create_table(db, cfg, sample_rows=rows)
    # Delete existing rows for the affected paths first (per-file replace semantics).
    paths = {c.path for c in chunks}
    if paths:
        path_list = ",".join(_sql_quote(p) for p in paths)
        try:
            tbl.delete(f"path IN ({path_list})")
        except Exception as e:  # pragma: no cover
            log.debug("delete-before-upsert failed (likely empty table): %s", e)
    tbl.add(rows)
    return len(rows)


def delete_for_path(cfg: Config, path: str) -> int:
    db = open_db(cfg)
    name = cfg.lancedb.table
    if name not in db.table_names():
        return 0
    tbl = db.open_table(name)
    try:
        tbl.delete(f"path = {_sql_quote(path)}")
    except Exception as e:  # pragma: no cover
        log.debug("delete_for_path %s: %s", path, e)
        return 0
    return 1


def search(
    cfg: Config,
    vector: list[float],
    k: int = 10,
    lang: str | None = None,
    path_prefix: str | None = None,
) -> list[dict[str, Any]]:
    db = open_db(cfg)
    name = cfg.lancedb.table
    if name not in db.table_names():
        return []
    tbl = db.open_table(name)
    q = tbl.search(vector).limit(k)
    filters: list[str] = []
    if lang:
        filters.append(f"lang = {_sql_quote(lang)}")
    if path_prefix:
        # Escape apostrophes then append the LIKE wildcard outside the literal.
        filters.append(f"path LIKE {_sql_quote(path_prefix + '%')}")
    if filters:
        q = q.where(" AND ".join(filters))
    df = q.to_list()
    return list(df)


def lookup_existing_hashes(cfg: Config, paths: set[str]) -> dict[tuple[str, str, int], str]:
    """Return ``{(path, symbol, start_line): content_hash}`` for rows whose
    ``path`` is in the given set.

    Used by the incremental indexer to skip embedding for chunks whose
    content_hash already matches what's in the DB. Paths not in the DB simply
    return no entries (caller treats as a full re-embed).

    Returns an empty dict on cold start (no table yet) or on read failure.
    """
    if not paths:
        return {}
    db = open_db(cfg)
    name = cfg.lancedb.table
    if name not in db.table_names():
        return {}
    tbl = db.open_table(name)
    path_list = ",".join(_sql_quote(p) for p in paths)
    try:
        rows = (
            tbl.search()
            .where(f"path IN ({path_list})")
            .select(["path", "symbol", "start_line", "content_hash"])
            .limit(1_000_000)
            .to_list()
        )
    except Exception as e:  # pragma: no cover
        log.debug("lookup_existing_hashes failed: %s", e)
        return {}
    out: dict[tuple[str, str, int], str] = {}
    for r in rows:
        h = r.get("content_hash") or ""
        # Legacy rows pre-v0.1.4 may have empty hash. Treat as cache miss
        # by simply not registering them in the lookup dict.
        if not h:
            continue
        out[(r["path"], r["symbol"], int(r["start_line"]))] = h
    return out


def list_indexed_paths(cfg: Config) -> set[str]:
    db = open_db(cfg)
    name = cfg.lancedb.table
    if name not in db.table_names():
        return set()
    tbl = db.open_table(name)
    rows = tbl.search().select(["path"]).limit(1_000_000).to_list()
    return {r["path"] for r in rows}


def table_stats(cfg: Config) -> dict[str, Any]:
    db = open_db(cfg)
    name = cfg.lancedb.table
    if name not in db.table_names():
        return {"rows": 0, "table": name}
    tbl = db.open_table(name)
    return {"rows": tbl.count_rows(), "table": name}


def ensure_dir(cfg: Config) -> Path:
    p = cfg.lancedb_path
    p.mkdir(parents=True, exist_ok=True)
    return p
