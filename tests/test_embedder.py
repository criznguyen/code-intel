"""Embedder unit tests. Mocks HTTP — no live Ollama in CI."""

from __future__ import annotations

import httpx
import pytest
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


def test_get_provider_voyage_raises_extension_boundary() -> None:
    """voyage is not shipped in core — must raise a helpful ValueError pointing
    to the plugin extension path. Bypasses pydantic Literal validation to
    exercise the runtime registry guard directly."""
    cfg = default_config(project_name="t")
    # object.__setattr__ bypasses pydantic's frozen/validated assignment so we
    # can construct an "out of band" provider name and verify the registry
    # guard is the one that raises (not pydantic itself).
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
