"""LanceDB read/write helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from code_intel._logging import get_logger
from code_intel.chunker import Chunk
from code_intel.config import Config

log = get_logger(__name__)


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


def open_db(cfg: Config):
    """Lazy import lancedb; open DB at configured path."""
    import lancedb  # type: ignore

    db_path = cfg.lancedb_path
    db_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


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
        path_list = ",".join(f"'{p}'" for p in paths)
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
        tbl.delete(f"path = '{path}'")
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
        filters.append(f"lang = '{lang}'")
    if path_prefix:
        filters.append(f"path LIKE '{path_prefix}%'")
    if filters:
        q = q.where(" AND ".join(filters))
    df = q.to_list()
    return list(df)


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
