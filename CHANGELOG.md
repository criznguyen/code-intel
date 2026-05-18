# Changelog

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
