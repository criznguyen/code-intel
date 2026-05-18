"""Embedder unit tests. Mocks HTTP — no live Ollama in CI."""

from __future__ import annotations

import httpx
import respx

from code_intel.config import default_config
from code_intel.embedder import OllamaProvider, get_provider


@respx.mock
def test_ollama_provider_embeds() -> None:
    cfg = default_config(project_name="t")
    provider = OllamaProvider(cfg.embedding)
    route = respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )
    out = provider.embed(["hello", "world"])
    assert route.call_count == 2
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_get_provider_default() -> None:
    cfg = default_config(project_name="t")
    p = get_provider(cfg)
    assert p.name == "ollama"
    assert p.dim == 768


def test_get_provider_unknown() -> None:
    cfg = default_config(project_name="t")
    cfg.embedding.provider = "nope"
    try:
        get_provider(cfg)
    except ValueError as e:
        assert "Unknown" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
