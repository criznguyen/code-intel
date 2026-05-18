"""Chunker behavior on tiny embedded source samples."""

from __future__ import annotations

import pytest

from code_intel.chunker import DEFAULT_MAX_CHUNK_CHARS, chunk_text

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


# ---------------------------------------------------------------------------
# v0.1.2: max_chunk_chars contract — never exceed cap, split (not drop).
# ---------------------------------------------------------------------------


def test_chunks_never_exceed_default_cap() -> None:
    """No emitted chunk may exceed DEFAULT_MAX_CHUNK_CHARS."""
    _try_or_skip("python")
    # Synthesize a giant function body well above 2500 chars.
    body_lines = "\n".join(f"    x_{i} = {i}" for i in range(400))
    giant = f"def huge():\n{body_lines}\n    return 0\n"
    chunks = chunk_text("huge.py", "python", giant)
    assert chunks, "should not drop the whole file"
    for c in chunks:
        assert len(c.content) <= DEFAULT_MAX_CHUNK_CHARS, (
            f"chunk {c.symbol} exceeded cap: {len(c.content)} > {DEFAULT_MAX_CHUNK_CHARS}"
        )


def test_chunks_respect_custom_cap() -> None:
    """Passing a smaller cap produces strictly smaller chunks."""
    _try_or_skip("python")
    body_lines = "\n".join(f"    x_{i} = {i}" for i in range(400))
    giant = f"def huge():\n{body_lines}\n    return 0\n"

    big = chunk_text("huge.py", "python", giant, max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS)
    small = chunk_text("huge.py", "python", giant, max_chunk_chars=1000)

    for c in small:
        assert len(c.content) <= 1000

    # Smaller cap must produce at least as many chunks (usually more).
    assert len(small) >= len(big)
    # And at least one chunk must exist (no silent drop).
    assert small


def test_oversized_chunk_split_preserves_lines() -> None:
    """Split sub-chunks should be tagged with `:partN` so callers can dedupe."""
    _try_or_skip("python")
    body_lines = "\n".join(f"    x_{i} = {i}" for i in range(400))
    giant = f"def huge():\n{body_lines}\n    return 0\n"
    chunks = chunk_text("huge.py", "python", giant, max_chunk_chars=500)
    part_chunks = [c for c in chunks if ":part" in c.symbol]
    assert part_chunks, "expected split sub-chunks tagged with :partN"


def test_markdown_oversized_section_splits() -> None:
    """Oversized markdown sections split at line boundaries, not dropped."""
    body = "\n".join(f"line {i} with enough text to push us over" for i in range(200))
    md = f"# Big Section\n\n{body}\n"
    chunks = chunk_text("big.md", "markdown", md, max_chunk_chars=400)
    assert chunks
    for c in chunks:
        assert len(c.content) <= 400
