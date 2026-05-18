# Changelog

## [0.1.5] - 2026-05-18

Closes the 4 residual items from the v0.1.4 audit on solanabot.

### Fixed

- **chunker** (MED-4 residual): markdown chunker now drops heading-only sections
  (e.g. `## Date: 2026-03-28` followed immediately by `## Version: 1.0`).
  v0.1.4 still emitted 13 such 1-line "section" chunks on the solanabot
  corpus — pure retrieval noise with no body content. The chunker treats a
  section as heading-only when its single non-blank line is itself a heading;
  files composed entirely of such sections still fall through to the
  whole-file fallback so they remain searchable.
- **search** (LOW-9 follow-up): `EmbeddingProvider` instances are now cached
  per `(provider, endpoint, model, dim, batch_size, timeout)` tuple. Previously
  every `semantic_search` call constructed a fresh `OllamaProvider` (and its
  `httpx.Client`), discarding keep-alive connections. Cache invalidates
  automatically on any config field change.

### Added

- **indexer** (INFO-11 part B): content-hash dedup for `--since` incremental
  re-index. New helper `store.lookup_existing_hashes(cfg, paths)` returns
  `{(path, symbol, start_line) -> content_hash}` for stored rows. The
  indexer partitions chunks into cache hits (unchanged content) and misses
  (changed / new), embedding only the misses. On a no-op `--since HEAD`
  (no file changes) the embed step is skipped entirely. `code-intel index
  --force` flag bypasses the cache (use after changing embedding model / dim).
  Closes ADR-0001.
- **search** reranker: cheap heuristic post-rerank over the top `k * 3`
  LanceDB candidates. Boosts `kind in (function, method)` (factor 0.85),
  penalizes `test_*` / `Test*` symbols (factor 1.15), boosts symbol↔query
  token overlap (factor 0.88 for ≥2 tokens, 0.94 for ≥1), and rewards
  *symbol coverage* — short symbols whose tokens are mostly in the query
  get an extra 0.85× (cov ≥0.9) or 0.92× (cov ≥0.5) factor. On solanabot
  `calculate token2022 transfer fee`, lifts `calculate_fee` from raw #17
  to reranked #2 (`transfer_fee` at #1), without disturbing top results
  for general queries like `clmm swap` or `jito tip`. Disable via
  `semantic_search(..., rerank=False)`.
- 14 new regression tests in `tests/test_v0_1_5_fixes.py`.

### Notes

- Perf: v0.1.4 measured p50 was 167.2ms vs v0.1.3 149.8ms. Profiling on
  solanabot shows Ollama embed dominates at ~130ms / call; remaining
  ~30-40ms is LanceDB ANN + result hydrate. The "regression" is Ollama-side
  load variance, not a code regression — the v0.1.4 `lancedb.connect` cache
  works (verified 0 reconnects across 5 queries). The provider cache added
  here is for clean architecture and ~5-10ms keep-alive savings.

## [0.1.4] - 2026-05-18

Closes all 11 findings from the v0.1.3 audit.

### Fixed (HIGH)

- **embedder** (HIGH-1): `OllamaProvider._embed_one` now refuses zero-length / dim-mismatch vectors so an empty Ollama response on whitespace prompts no longer corrupts LanceDB. `search.semantic_search` rejects empty/whitespace queries with `ValueError("query must be non-empty")` instead of letting them propagate.
- **chunker** (HIGH-2): `_whole_file_chunk` fallback now routes through `_split_oversized` when the head section exceeds `max_chunk_chars`. Previously emitted a single 200-line chunk that was silently truncated downstream.

### Fixed (MEDIUM)

- **chunker** (MED-3): Rust `pub mod foo;` / `mod foo;` forward declarations are filtered before chunk emission so 1-line decl junk no longer pollutes top-K. Inline `mod foo { ... }` bodies (with braces) are still chunked.
- **chunker** (MED-4): Markdown chunker now state-tracks fenced code blocks (` ``` ` and `~~~`). Shell-comment lines like `# Setup` inside fenced code blocks are no longer mis-promoted to H1 sections.
- **store** (MED-5): All LanceDB SQL string filters route through `_sql_quote()` which doubles embedded apostrophes. Re-indexing a file like `src/it's.rs` is now a clean upsert instead of a duplicate-row append + silent parse error.
- **watcher / cli** (MED-6): `watcher.watch` now distinguishes `Change.deleted` events from add/modify and routes each to `store.delete_for_path` so orphan chunks are removed on file delete/rename. `code-intel index --prune` flag added — discovers current files, compares against indexed paths, deletes the diff.

### Fixed (LOW)

- **embedder** (LOW-7): hardcoded 60s HTTP timeout is now `[embedding].timeout_seconds` (default 60.0). Slow CPUs / large embed models can raise this without forking.
- **embedder** (LOW-8): `cfg.embedding.batch_size` is now honored — `OllamaProvider.embed` tries `/api/embed` with `input: [str]` per batch and falls back permanently (sticky bit per-instance) to per-item `/api/embeddings` on 404. Old Ollama (< 0.2) keeps working; new Ollama gets 1 POST per batch instead of N.
- **store** (LOW-9): `open_db` caches the LanceDB connection per `(resolved_path, table)` behind a `threading.Lock`. Search hot path no longer pays ~10 ms + allocation pressure per call. Test helper `_reset_db_cache()` exposed for tmp-path test isolation.

### Fixed (INFO)

- **indexer** (INFO-11 part A): `_walk_repo` now uses `os.walk` with in-place `dirs[:]` pruning honoring `exclude_globs`, so `target/`, `node_modules/`, etc. are no longer descended into. Saves ~5-10 s on large repos.
- **bench** (INFO-10): added `scripts/bench_memory.py` to measure RSS delta over N searches. Reference point: v0.1.3 leaked ~650 MB across 50 queries due to per-call DB connection; LOW-9 cache materially closes that gap.

### Deferred

- **INFO-11 part B** (content-hash dedup for `--since`): documented as `docs/adr/0001-content-hash-dedup.md`, scheduled for v0.1.5. `--since HEAD` is still O(changed-files × embed-cost) in v0.1.4.

### Added

- `[embedding].timeout_seconds` config field.
- `code-intel index --prune` flag.
- `scripts/bench_memory.py`.
- 18 new regression tests in `tests/test_v0_1_4_fixes.py` — one per finding plus contract pins for `_sql_quote`, `_RUST_MOD_DECL_RE`, `_whole_file_chunk` return shape.

## [0.1.3] - 2026-05-18

### Fixed
- `search.py` failed with "cannot unpack non-iterable EmbedResult object" — caller missed during v0.1.2 embedder return-type refactor. Updated to use `.vectors` and raise `RuntimeError` with `skipped_reasons` if the query itself failed to embed.

## [0.1.2] - 2026-05-18

### Fixed
- **chunker**: lowered default `max_chunk_chars` from 8000 → 2500 to fit the 2048-token context window of common Ollama embedding models (`embeddinggemma`, `nomic-embed-text`, `mxbai-embed-large`). Now configurable via `[index].max_chunk_chars` in `config.toml`. Oversized syntactic units are split at line boundaries (suffix `:partN`) instead of silently dropped.
- **embedder**: per-chunk Ollama HTTP 4xx/5xx and transport errors no longer abort the full index run. `OllamaProvider.embed(...)` now returns an `EmbedResult(vectors, skipped_indices, skipped_reasons)`; the indexer surfaces `skipped=K` in the summary and writes per-chunk audit lines to `.codeindex/skipped.jsonl`.
- **watcher**: added `__main__` block so `python -m code_intel.watcher <target>` actually runs the daemon instead of importing-and-exiting-0.
- **systemd**: `install-services` now resolves `sys.executable` + `shutil.which("code-intel")` at install time and substitutes them into `ExecStart`, so the watcher and MCP units work under systemd user sessions (which lack pyenv shims). Added `--force` flag to overwrite stale unit files. Project manifest at `~/.config/code-intel/projects/<instance>.toml` remains the source of truth for the target path.
- **mcp-config**: `--scope project` now emits the entry under key `"code-intel"` (project-scoped `.mcp.json` is already path-namespaced by Claude Code). `--scope user` keeps `"code-intel-<project>"` to avoid collisions across multiple projects sharing `~/.claude.json`.

### Added
- `IndexSection.max_chunk_chars` config field (default 2500).
- `EmbedResult` dataclass exposing per-chunk skip metadata.
- `.codeindex/skipped.jsonl` audit log for embedder failures.
- `install-services --force` flag.
- Tests: chunk-size cap contract, custom-cap override, skip-on-5xx/transport, install-services renders absolute interpreter path, install-services respects existing units without `--force`.

## [0.1.1] - 2026-05-18

### Removed
- `VoyageProvider` and `OpenAIProvider` embedding stubs that raised `NotImplementedError`. code-intel core is local-first via Ollama; paid-API providers should be implemented as separate plugin packages using the `EmbeddingProvider` Protocol.

### Changed
- `[embedding].provider` config field now accepts only `"ollama"` in core. Extension packages can register their own providers via the Protocol interface.

## [0.1.0] - 2026-05-18

Initial release.
