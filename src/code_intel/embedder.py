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
        self._client = httpx.Client(timeout=cfg.timeout_seconds)
        # Sticky bit: set after the first /api/embed 404 so subsequent batches
        # skip the round-trip. Per-instance so tests stay isolated.
        self._batch_endpoint_disabled: bool = False

    def _embed_one(self, text: str) -> list[float]:
        # Ollama exposes both /api/embeddings (legacy) and /api/embed (newer).
        # Use /api/embeddings (single-prompt) for max compatibility.
        url = f"{self.endpoint}/api/embeddings"
        resp = self._client.post(url, json={"model": self.model, "prompt": text})
        resp.raise_for_status()
        body = resp.json()
        if "embedding" not in body:
            raise RuntimeError(f"Ollama response missing 'embedding': {body}")
        vec = list(body["embedding"])
        # Ollama returns HTTP 200 with `embedding: []` on whitespace-only prompts.
        # Refuse zero-length / dim-mismatch vectors so LanceDB doesn't corrupt
        # the table or NEAREST search downstream (HIGH-1 in v0.1.3 audit).
        if len(vec) != self.dim:
            raise RuntimeError(
                f"Ollama returned vector of length {len(vec)}, expected {self.dim} "
                f"(prompt chars={len(text)}, likely empty/whitespace input)"
            )
        return vec

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """POST to /api/embed (Ollama >= 0.2) with `input: [str]`.

        Raises ``NotImplementedError`` on 404 so the caller can fall back to
        the per-item /api/embeddings legacy endpoint.
        """
        if self._batch_endpoint_disabled:
            raise NotImplementedError("legacy ollama, /api/embed not available")
        url = f"{self.endpoint}/api/embed"
        resp = self._client.post(url, json={"model": self.model, "input": texts})
        if resp.status_code == 404:
            self._batch_endpoint_disabled = True
            raise NotImplementedError("ollama returned 404 for /api/embed")
        resp.raise_for_status()
        body = resp.json()
        embs = body.get("embeddings")
        if not isinstance(embs, list) or len(embs) != len(texts):
            raise RuntimeError(
                f"Ollama /api/embed returned malformed batch: "
                f"got {type(embs).__name__} len={len(embs) if isinstance(embs, list) else '?'} "
                f"for {len(texts)} inputs"
            )
        out: list[list[float]] = []
        for i, v in enumerate(embs):
            if not isinstance(v, list) or len(v) != self.dim:
                raise RuntimeError(
                    f"Ollama /api/embed item {i}: vector dim={len(v) if isinstance(v, list) else '?'}, "
                    f"expected {self.dim} (input chars={len(texts[i])})"
                )
            out.append(list(v))
        return out

    def embed(self, texts: list[str]) -> EmbedResult:
        result = EmbedResult()
        batch_size = max(1, self.batch_size)
        total = len(texts)
        # v0.1.6 NEW-1: progress log every batch so a foreground `--full`
        # reindex on a large repo (20+ min) isn't visually frozen. Logged at
        # INFO so the CLI auto-INFO elevation surfaces it without `-v`.
        # We log on entry-to-batch (with done-so-far counter) rather than
        # on exit so the user sees activity even when an individual batch
        # is the slow one.
        n_batches = (total + batch_size - 1) // batch_size
        # We chunk inputs into batches and try the batch endpoint first. On
        # batch-level failure we fall back to per-item _embed_one so a single
        # bad chunk in a batch doesn't take down the surrounding survivors.
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1
            if total > batch_size:
                # Skip log on single-batch calls (query embed at search time).
                log.info(
                    "ollama embed batch %d/%d (%d/%d items)",
                    batch_idx,
                    n_batches,
                    batch_start,
                    total,
                )
            try:
                vecs = self._embed_batch(batch)
                for i, v in enumerate(vecs):
                    idx = batch_start + i
                    result.vectors.append(v)
                continue
            except NotImplementedError:
                # /api/embed not available; permanent fallback (sticky bit set).
                pass
            except (httpx.HTTPStatusError, httpx.RequestError, RuntimeError) as e:
                log.debug(
                    "ollama batch embed fell back to per-item (batch_start=%d size=%d): %s",
                    batch_start,
                    len(batch),
                    e,
                )
            # Per-item fallback (legacy /api/embeddings, also used for batch failure).
            for i, t in enumerate(batch):
                idx = batch_start + i
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
