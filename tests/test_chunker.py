"""Chunker behavior on tiny embedded source samples."""

from __future__ import annotations

import pytest

from code_intel.chunker import chunk_text

PYTHON_SAMPLE = '''
def add(a: int, b: int) -> int:
    """Add two ints."""
    return a + b


class Calc:
    def mul(self, x, y):
        return x * y
'''.lstrip()


RUST_SAMPLE = """
pub fn hello() -> &'static str {
    "world"
}

struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Self { Self { count: 0 } }
}
""".lstrip()


MARKDOWN_SAMPLE = """
# Title

Top paragraph.

## Section One

First section content.

## Section Two

Second section content.
""".lstrip()


def _try_or_skip(lang: str):
    """If tree-sitter parser cannot load for `lang`, skip the test."""
    from code_intel.chunker import _get_parser

    if _get_parser(lang) is None:  # pragma: no cover - env-dependent
        pytest.skip(f"tree-sitter parser unavailable for {lang}")


def test_chunk_python() -> None:
    _try_or_skip("python")
    chunks = chunk_text("sample.py", "python", PYTHON_SAMPLE)
    symbols = {c.symbol for c in chunks}
    kinds = {c.kind for c in chunks}
    assert "add" in symbols
    assert "Calc" in symbols
    assert "mul" in symbols  # nested method captured too
    assert "function" in kinds
    assert "class" in kinds


def test_chunk_rust() -> None:
    _try_or_skip("rust")
    chunks = chunk_text("sample.rs", "rust", RUST_SAMPLE)
    symbols = {c.symbol for c in chunks}
    assert "hello" in symbols
    assert "Counter" in symbols


def test_chunk_markdown_sections() -> None:
    chunks = chunk_text("doc.md", "markdown", MARKDOWN_SAMPLE)
    titles = {c.symbol for c in chunks}
    assert "Title" in titles
    assert "Section One" in titles
    assert "Section Two" in titles
    for c in chunks:
        assert c.kind == "section"
        assert c.lang == "markdown"
