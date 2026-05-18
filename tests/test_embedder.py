"""Embedder unit tests. Mocks HTTP — no live Ollama in CI."""

from __future__ import annotations

import httpx
import pytest
import respx

from code_intel.config import default_config
from code_intel.embedder import EmbedResult, OllamaProvider, get_provider


@respx.mock
def test_ollama_provider_embeds() -> None:
    cfg = default_config(project_name="t")
    provider = OllamaProvider(cfg.embedding)
    route = respx.post("http://localhost:11434/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})
    )
    result = provider.embed(["hello", "world"])
    assert route.call_count == 2
    assert isinstance(result, EmbedResult)
    assert result.vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert result.skipped_indices == []


@respx.mock
def test_ollama_provider_skips_5xx_on_single_chunk() -> None:
    """500 on chunk 3 of 5: survivors embed, skipped count surfaces, no abort."""
    cfg = default_config(project_name="t")
    provider = OllamaProvider(cfg.embedding)
    ok = httpx.Response(200, json={"embedding": [0.42, 0.42]})
    bad = httpx.Response(500, json={"error": "model too long"})
    # respx returns side_effect responses in order across calls.
    respx.post("http://localhost:11434/api/embeddings").mock(side_effect=[ok, ok, bad, ok, ok])
    texts = ["a", "b", "c-giant", "d", "e"]
    result = provider.embed(texts)

    # Survivors: chunks 0, 1, 3, 4 → 4 vectors total.
    assert len(result.vectors) == 4
    assert result.skipped_indices == [2]
    assert "http_500" in result.skipped_reasons[2]


@respx.mock
def test_ollama_provider_skips_transport_error() -> None:
    cfg = default_config(project_name="t")
    provider = OllamaProvider(cfg.embedding)
    ok = httpx.Response(200, json={"embedding": [1.0]})
    respx.post("http://localhost:11434/api/embeddings").mock(
        side_effect=[ok, httpx.ConnectError("boom"), ok]
    )
    result = provider.embed(["a", "b", "c"])
    assert len(result.vectors) == 2
    assert result.skipped_indices == [1]
    assert "transport" in result.skipped_reasons[1]


def test_get_provider_default() -> None:
    cfg = default_config(project_name="t")
    p = get_provider(cfg)
    assert p.name == "ollama"
    assert p.dim == 768


def test_get_provider_voyage_raises_extension_boundary() -> None:
    """voyage is not shipped in core — must raise a helpful ValueError pointing
    to the plugin extension path. Bypasses pydantic Literal validation to
    exercise the runtime registry guard directly."""
    cfg = default_config(project_name="t")
    object.__setattr__(cfg.embedding, "provider", "voyage")
    with pytest.raises(ValueError) as excinfo:
        get_provider(cfg)
    msg = str(excinfo.value)
    assert "voyage" in msg.lower()
    assert "plugin" in msg.lower()


def test_get_provider_openai_raises_extension_boundary() -> None:
    """Same guard for openai."""
    cfg = default_config(project_name="t")
    object.__setattr__(cfg.embedding, "provider", "openai")
    with pytest.raises(ValueError) as excinfo:
        get_provider(cfg)
    assert "openai" in str(excinfo.value).lower()


def test_get_provider_unknown() -> None:
    cfg = default_config(project_name="t")
    object.__setattr__(cfg.embedding, "provider", "nope")
    with pytest.raises(ValueError) as excinfo:
        get_provider(cfg)
    assert "nope" in str(excinfo.value).lower()
