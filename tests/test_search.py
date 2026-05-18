"""Unit tests for semantic_search query-embedding path.

Regression guard for v0.1.3: search.py previously did `[vec] = provider.embed([query])`,
which broke when v0.1.2 changed the return type from `list[list[float]]` to
`EmbedResult`. These tests pin the new contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from code_intel.config import default_config
from code_intel.embedder import EmbedResult
from code_intel.search import _reset_provider_cache, semantic_search


class _StubProvider:
    name = "stub"
    dim = 768

    def __init__(self, result: EmbedResult) -> None:
        self._result = result
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbedResult:
        self.calls.append(list(texts))
        return self._result


def test_semantic_search_unpacks_vectors_field(tmp_path) -> None:
    """Happy path: provider returns vectors, store.search is called with [0]."""
    _reset_provider_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path

    fake_vec = [0.0] * 768
    provider = _StubProvider(
        EmbedResult(vectors=[fake_vec], skipped_indices=[], skipped_reasons={})
    )

    fake_rows = [
        {
            "path": "src/foo.py",
            "symbol": "foo",
            "kind": "function",
            "lang": "python",
            "start_line": 1,
            "end_line": 10,
            "content": "def foo(): ...",
            "_distance": 0.123,
        }
    ]

    with (
        patch("code_intel.search.get_provider", return_value=provider),
        patch("code_intel.store.search", return_value=fake_rows) as db_search,
    ):
        # rerank=False so this test pins the raw-vector contract (no
        # overfetch / no reorder). The rerank=True path is tested separately
        # in test_v0_1_5_fixes.py.
        results = semantic_search(cfg, "what is foo", k=5, rerank=False)

    assert len(results) == 1
    assert results[0]["path"] == "src/foo.py"
    assert results[0]["symbol"] == "foo"
    assert results[0]["score"] == 0.123
    # store.search received the vector we extracted from `result.vectors[0]`.
    args, kwargs = db_search.call_args
    assert args[1] == fake_vec
    assert kwargs.get("k") == 5
    # provider.embed was called exactly once with the query wrapped in a list.
    assert provider.calls == [["what is foo"]]
    _reset_provider_cache()


def test_semantic_search_raises_when_query_embed_skipped(tmp_path) -> None:
    """Provider skipped the query itself → RuntimeError with the skip reason."""
    _reset_provider_cache()
    cfg = default_config(project_name="t")
    cfg._target = tmp_path

    provider = _StubProvider(
        EmbedResult(vectors=[], skipped_indices=[0], skipped_reasons={0: "test"})
    )

    with (
        patch("code_intel.search.get_provider", return_value=provider),
        pytest.raises(RuntimeError) as excinfo,
    ):
        semantic_search(cfg, "any query", k=5)

    assert "test" in str(excinfo.value)
    _reset_provider_cache()
