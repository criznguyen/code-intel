"""Embedding provider plugin system.

code-intel core is local-first: the only provider shipped in this package is
:class:`OllamaProvider`. The :class:`EmbeddingProvider` Protocol is public so
external plugin packages can implement and register their own providers
(Voyage, OpenAI, Cohere, etc.) without forking this repo.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from code_intel._logging import get_logger
from code_intel.config import Config, EmbeddingSection

log = get_logger(__name__)


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


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

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for t in batch:
                out.append(self._embed_one(t))
        return out


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
