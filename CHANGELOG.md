# Changelog

## [0.1.1] - 2026-05-18

### Removed
- `VoyageProvider` and `OpenAIProvider` embedding stubs that raised `NotImplementedError`. code-intel core is local-first via Ollama; paid-API providers should be implemented as separate plugin packages using the `EmbeddingProvider` Protocol.

### Changed
- `[embedding].provider` config field now accepts only `"ollama"` in core. Extension packages can register their own providers via the Protocol interface.

## [0.1.0] - 2026-05-18

Initial release.
