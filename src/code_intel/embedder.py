"""Embedding provider plugin system.

code-intel core is local-first: the only provider shipped in this package is
:class:`OllamaProvider`. The :class:`EmbeddingProvider` Protocol is public so
external plugin packages can implement and register their own providers
(Voyage, OpenAI, Cohere, etc.) without forking this repo.

Per-chunk failures are logged + skipped instead of aborting the whole run:
:class:`EmbedResult` carries the surviving vectors plus the indices of any
skipped chunks so the caller can record + audit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from code_intel._logging import get_logger
from code_intel.config import Config, EmbeddingSection

log = get_logger(__name__)


@dataclass
class EmbedResult:
    """Outcome of an `embed(texts)` call.

    `vectors` is shorter than `texts` when chunks were skipped. The original
    1:1 mapping is recoverable via `skipped_indices` (sorted, ascending).
    """

    vectors: list[list[float]] = field(default_factory=list)
    skipped_indices: list[int] = field(default_factory=list)
    skipped_reasons: dict[int, str] = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> EmbedResult: ...


class OllamaProvider:
    name = "ollama"

    def __init__(self, cfg: EmbeddingSection):
        self.endpoint = cfg.endpoint.rstrip("/")
        self.model = cfg.model
        self.batch_size = cfg.batch_size
        self.dim = cfg.dim
        self._client = httpx.Client(timeout=60.0)

    def _embed_one(self, text: str) -> list[float]:
        # Ollama exposes both /api/embeddings (legacy) and /api/embed (newer).
        # Use /api/embeddings (single-prompt) for max compatibility.
        url = f"{self.endpoint}/api/embeddings"
        resp = self._client.post(url, json={"model": self.model, "prompt": text})
        resp.raise_for_status()
        body = resp.json()
        if "embedding" not in body:
            raise RuntimeError(f"Ollama response missing 'embedding': {body}")
        return list(body["embedding"])

    def embed(self, texts: list[str]) -> EmbedResult:
        result = EmbedResult()
        for idx, t in enumerate(texts):
            try:
                result.vectors.append(self._embed_one(t))
            except httpx.HTTPStatusError as e:
                reason = f"http_{e.response.status_code}: {e.response.text[:200]!r}"
                log.warning(
                    "ollama embed skip idx=%d chars=%d status=%s reason=%s",
                    idx,
                    len(t),
                    e.response.status_code,
                    reason,
                )
                result.skipped_indices.append(idx)
                result.skipped_reasons[idx] = reason
            except httpx.RequestError as e:
                reason = f"transport: {e!r}"
                log.warning(
                    "ollama embed skip idx=%d chars=%d transport-error=%s",
                    idx,
                    len(t),
                    e,
                )
                result.skipped_indices.append(idx)
                result.skipped_reasons[idx] = reason
            except RuntimeError as e:
                reason = f"protocol: {e}"
                log.warning(
                    "ollama embed skip idx=%d chars=%d protocol-error=%s",
                    idx,
                    len(t),
                    e,
                )
                result.skipped_indices.append(idx)
                result.skipped_reasons[idx] = reason
        return result


_REGISTRY: dict[str, type] = {
    "ollama": OllamaProvider,
}


def get_provider(cfg: Config) -> EmbeddingProvider:
    name = cfg.embedding.provider.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unsupported embedding provider: {name!r}. "
            "Only 'ollama' is shipped in core; add a plugin package to extend."
        )
    return _REGISTRY[name](cfg.embedding)
