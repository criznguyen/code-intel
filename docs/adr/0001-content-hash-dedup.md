# ADR-0001: content-hash dedup for `--since` re-index

- **Status**: Proposed
- **Date**: 2026-05-18
- **Scope**: code-intel v0.1.5

## Context

INFO-11 in the v0.1.3 audit flagged that `code-intel index --since HEAD` is
not idempotent: every changed file is unconditionally re-chunked, re-embedded,
and re-upserted, even when the chunk content hash is byte-identical to what
LanceDB already stores. On large repos this is the difference between a
~50 ms incremental and a 4-minute re-embed of unchanged files.

The chunk schema already carries `content_hash` (BLAKE3 or SHA-256
fallback), so the DB has the information needed; the indexer just does not
consult it before queueing chunks for embedding.

## Decision

Defer the optimization to **v0.1.5**. Scope:

1. Before `provider.embed(texts)`, batch-lookup `content_hash` for the set
   of `(path, symbol, start_line)` triples already in LanceDB.
2. Partition chunks into `hit` (hash matches existing row) and `miss`.
3. Embed only `miss`; for `hit`, reuse the stored vector via a
   `tbl.search().where(...).select(["vector"])` round-trip, then upsert
   with the existing vector unchanged. (Or skip the upsert entirely when
   `id` is byte-identical.)
4. CLI: add `--force` flag to bypass the cache when the user changed
   embedding model or `dim`.

## Why not in v0.1.4

- v0.1.4 closes 11 audit findings; adding a new contract (hash-based
  cache lookup) on top of those would blur regression attribution.
- The fix requires a new helper `store.lookup_existing_hashes(cfg, ids) -> dict[id, hash]`
  that touches the LanceDB query layer — non-trivial and worth its own
  audit pass.
- INFO-11 part A (rglob → os.walk + prune) is shipping in v0.1.4 and
  already cuts ~5-10 s of discovery overhead on solanabot-sized repos,
  which is the immediate user pain point.

## Consequences

- `--since HEAD` remains O(changed-files × embed-cost) until v0.1.5.
  Documented as a known limitation in the v0.1.4 CHANGELOG.
- No schema migration required when v0.1.5 lands — the column already exists.

## Open questions for v0.1.5

- Cache invalidation on `embedding.model` change: should the table track
  the model id per row (additional column) or rely on operator running
  `code-intel index --force`?
- LanceDB row-level update API support — falls back to delete+add if
  not available.
