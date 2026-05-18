"""Tree-sitter AST chunker. Extracts function/class/section chunks from source files."""

from __future__ import annotations

import hashlib
import importlib
import re
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from code_intel._logging import get_logger

log = get_logger(__name__)

# Matches Rust forward `mod foo;` declarations (no body). Inline `mod x { … }`
# always contains `{`, so this regex excludes them.
_RUST_MOD_DECL_RE = re.compile(r"^\s*(pub(\s*\([^)]*\))?\s+)?mod\s+\w+\s*;\s*$")


# Default maximum chunk size in characters. Conservative value chosen to fit
# the 2048-token context window of common Ollama embedding models
# (embeddinggemma, nomic-embed-text, mxbai-embed-large) for dense code at
# ~4-5 chars per token. Override via `[index].max_chunk_chars` in config.toml.
#
# Historical note: v0.1.0/v0.1.1 hardcoded 8000, which silently dropped any
# chunk above the cap AND produced HTTP 500 from Ollama on chunks that did
# slip through. v0.1.2 lowers default to 2500 and splits at line boundaries
# instead of dropping.
DEFAULT_MAX_CHUNK_CHARS = 2500
# For files with no parseable symbols, take this many leading lines as a single chunk.
WHOLE_FILE_FALLBACK_LINES = 200


class Chunk(BaseModel):
    path: str  # repo-relative
    lang: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    content_hash: str


# Extension -> internal language name.
EXT_TO_LANG: dict[str, str] = {
    ".rs": "rust",
    ".py": "python",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
}

# Mapping: lang -> (pip-module-name, attr-on-module returning ts language ptr).
LANG_PACKAGE: dict[str, tuple[str, str]] = {
    "rust": ("tree_sitter_rust", "language"),
    "python": ("tree_sitter_python", "language"),
    "go": ("tree_sitter_go", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "javascript": ("tree_sitter_javascript", "language"),
    "java": ("tree_sitter_java", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
}

# Per-language node-type -> chunk kind and the field name carrying the symbol id.
LANG_NODE_RULES: dict[str, dict[str, dict[str, str | None]]] = {
    "rust": {
        "function_item": {"kind": "function", "name_field": "name"},
        "impl_item": {"kind": "impl", "name_field": None},
        "struct_item": {"kind": "class", "name_field": "name"},
        "enum_item": {"kind": "class", "name_field": "name"},
        "trait_item": {"kind": "class", "name_field": "name"},
        "mod_item": {"kind": "module", "name_field": "name"},
    },
    "python": {
        "function_definition": {"kind": "function", "name_field": "name"},
        "class_definition": {"kind": "class", "name_field": "name"},
    },
    "go": {
        "function_declaration": {"kind": "function", "name_field": "name"},
        "method_declaration": {"kind": "method", "name_field": "name"},
        "type_declaration": {"kind": "class", "name_field": None},
    },
    "javascript": {
        "function_declaration": {"kind": "function", "name_field": "name"},
        "class_declaration": {"kind": "class", "name_field": "name"},
        "method_definition": {"kind": "method", "name_field": "name"},
    },
    "typescript": {
        "function_declaration": {"kind": "function", "name_field": "name"},
        "class_declaration": {"kind": "class", "name_field": "name"},
        "method_definition": {"kind": "method", "name_field": "name"},
        "interface_declaration": {"kind": "class", "name_field": "name"},
    },
    "tsx": {
        "function_declaration": {"kind": "function", "name_field": "name"},
        "class_declaration": {"kind": "class", "name_field": "name"},
        "method_definition": {"kind": "method", "name_field": "name"},
        "interface_declaration": {"kind": "class", "name_field": "name"},
    },
    "java": {
        "method_declaration": {"kind": "method", "name_field": "name"},
        "class_declaration": {"kind": "class", "name_field": "name"},
        "interface_declaration": {"kind": "class", "name_field": "name"},
    },
    "kotlin": {
        "function_declaration": {"kind": "function", "name_field": "name"},
        "class_declaration": {"kind": "class", "name_field": "name"},
    },
}


def detect_lang(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".md":
        return "markdown"
    return EXT_TO_LANG.get(ext)


def _hash_content(text: str) -> str:
    try:
        import blake3

        return blake3.blake3(text.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _whole_file_chunk(
    rel_path: str,
    lang: str,
    text: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Fallback chunk for files with no parseable symbols.

    Returns a list (not a single chunk) so callers can splice. If the head
    section exceeds ``max_chunk_chars`` it is split at line boundaries via
    :func:`_split_oversized` rather than emitted whole and silently truncated
    by downstream embed-cap logic. (HIGH-2 in v0.1.3 audit.)
    """
    lines = text.splitlines()
    head = "\n".join(lines[:WHOLE_FILE_FALLBACK_LINES])
    symbol = Path(rel_path).name
    if len(head) > max_chunk_chars:
        return _split_oversized(
            rel_path=rel_path,
            lang=lang,
            symbol=symbol,
            kind="module",
            start_line=1,
            content=head,
            max_chunk_chars=max_chunk_chars,
        )
    return [
        Chunk(
            path=rel_path,
            lang=lang,
            symbol=symbol,
            kind="module",
            start_line=1,
            end_line=min(len(lines), WHOLE_FILE_FALLBACK_LINES),
            content=head,
            content_hash=_hash_content(head),
        )
    ]


def _split_oversized(
    rel_path: str,
    lang: str,
    symbol: str,
    kind: str,
    start_line: int,
    content: str,
    max_chunk_chars: int,
) -> list[Chunk]:
    """Split an oversized chunk into ≤ max_chunk_chars sub-chunks at line boundaries.

    Each sub-chunk gets a suffix `:partN` on the symbol so callers can tell them
    apart. start_line is computed for each sub-chunk relative to the parent.
    """
    if max_chunk_chars <= 0:
        return []
    lines = content.splitlines(keepends=True)
    sub_chunks: list[Chunk] = []
    part = 0
    buf: list[str] = []
    buf_len = 0
    line_offset = 0  # offset of buf[0] from start_line
    cur_line_offset = 0

    def _flush(local_offset: int) -> None:
        nonlocal part, buf, buf_len
        if not buf:
            return
        part += 1
        body = "".join(buf).rstrip("\n")
        sub_start = start_line + local_offset
        sub_end = sub_start + len(buf) - 1
        sub_chunks.append(
            Chunk(
                path=rel_path,
                lang=lang,
                symbol=f"{symbol}:part{part}",
                kind=kind,
                start_line=sub_start,
                end_line=sub_end,
                content=body,
                content_hash=_hash_content(body),
            )
        )
        buf = []
        buf_len = 0

    for line in lines:
        # Single line itself longer than the cap: hard-truncate that one line.
        if len(line) > max_chunk_chars:
            _flush(line_offset)
            part += 1
            trunc = line[:max_chunk_chars].rstrip("\n")
            trunc_body = trunc + f"\n# [code-intel: line truncated, original {len(line)} chars]"
            sub_start = start_line + cur_line_offset
            sub_chunks.append(
                Chunk(
                    path=rel_path,
                    lang=lang,
                    symbol=f"{symbol}:part{part}",
                    kind=kind,
                    start_line=sub_start,
                    end_line=sub_start,
                    content=trunc_body,
                    content_hash=_hash_content(trunc_body),
                )
            )
            cur_line_offset += 1
            line_offset = cur_line_offset
            continue
        if buf_len + len(line) > max_chunk_chars and buf:
            _flush(line_offset)
            line_offset = cur_line_offset
        buf.append(line)
        buf_len += len(line)
        cur_line_offset += 1
    _flush(line_offset)
    return sub_chunks


_MARKDOWN_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _chunk_markdown(rel_path: str, text: str, max_chunk_chars: int) -> list[Chunk]:
    """Split markdown by H1/H2 headings, then split oversized sections.

    Lines inside fenced code blocks (``` or ~~~) are *not* treated as headings
    even if they start with `#`. Shell scripts in markdown commonly contain
    `# Setup` comments which v0.1.3 mistakenly promoted to H1 sections.
    (MED-4 in v0.1.3 audit.)
    """
    lines = text.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    current_title = Path(rel_path).stem
    current_start = 1
    current_lines: list[str] = []
    in_fence = False

    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if _MARKDOWN_FENCE_RE.match(line):
            # Toggle fence state. Opening fence may have a language tag,
            # closing fence usually does not — but we just toggle either way.
            in_fence = not in_fence
            current_lines.append(line)
            continue
        if not in_fence and stripped.startswith(("# ", "## ")):
            if current_lines:
                sections.append((current_title, current_start, current_lines))
            current_title = stripped.lstrip("#").strip() or f"section-{i}"
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_start, current_lines))

    chunks: list[Chunk] = []
    for title, start, body in sections:
        content = "\n".join(body)
        # MED-4 residual (v0.1.5): drop heading-only sections.
        # Files with stacked metadata headings (`## Date: …`, `## Version: …`)
        # produce 1-line "section" chunks containing only the heading line and
        # no body. These are useless retrieval noise — they get embedded as
        # short title strings and dilute top-K. We treat a section as
        # heading-only when its non-blank lines == 1 and that line is itself
        # a heading. The very first chunk of a file (synthetic title before
        # the first explicit heading) is exempt — keep it as a file-anchor.
        non_blank = [ln for ln in body if ln.strip()]
        is_heading_only = len(non_blank) == 1 and non_blank[0].lstrip().startswith(("# ", "## "))
        if is_heading_only:
            continue
        if len(content) > max_chunk_chars:
            # Split oversized markdown section at line boundaries.
            chunks.extend(
                _split_oversized(
                    rel_path=rel_path,
                    lang="markdown",
                    symbol=title,
                    kind="section",
                    start_line=start,
                    content=content,
                    max_chunk_chars=max_chunk_chars,
                )
            )
            continue
        chunks.append(
            Chunk(
                path=rel_path,
                lang="markdown",
                symbol=title,
                kind="section",
                start_line=start,
                end_line=start + len(body) - 1,
                content=content,
                content_hash=_hash_content(content),
            )
        )
    if not chunks:
        # Either the file was empty/whitespace OR every section was
        # heading-only. Either way: fall through to whole-file fallback so
        # callers get a usable chunk (and `text.strip()` is empty path is
        # already short-circuited by `chunk_file` upstream).
        chunks.extend(_whole_file_chunk(rel_path, "markdown", text, max_chunk_chars))
    return chunks


@cache
def _get_parser(lang: str):
    """Lazy-load a tree-sitter parser for `lang`. Returns None on failure."""
    spec = LANG_PACKAGE.get(lang)
    if spec is None:
        return None
    module_name, attr = spec
    try:
        from tree_sitter import Language, Parser

        mod = importlib.import_module(module_name)
        ts_lang = Language(getattr(mod, attr)())
        return Parser(ts_lang)
    except Exception as e:  # pragma: no cover - env-dependent
        log.debug("tree-sitter parser unavailable for %s: %s", lang, e)
        return None


def _node_name(node, source: bytes, name_field: str | None) -> str:
    if name_field:
        named = node.child_by_field_name(name_field)
        if named is not None:
            return _safe_decode(source[named.start_byte : named.end_byte])
    # Fallback: first 'identifier' / 'type_identifier' child.
    for child in node.children:
        if child.type in {"identifier", "type_identifier", "field_identifier"}:
            return _safe_decode(source[child.start_byte : child.end_byte])
    return f"anon@{node.start_point[0] + 1}"


def _walk_tree(root, source: bytes, rules: dict) -> list[tuple[str, str, object]]:
    """Walk syntax tree collecting (symbol, kind, node) for nodes matching rules."""
    found: list[tuple[str, str, object]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        rule = rules.get(node.type)
        if rule is not None:
            symbol = _node_name(node, source, rule.get("name_field"))
            found.append((symbol, rule["kind"], node))
        # Descend regardless — supports nested defs (methods in classes etc.).
        stack.extend(node.children)
    return found


def _chunk_with_treesitter(
    rel_path: str, lang: str, text: str, max_chunk_chars: int
) -> list[Chunk]:
    rules = LANG_NODE_RULES.get(lang)
    if rules is None:
        return _whole_file_chunk(rel_path, lang, text, max_chunk_chars)

    parser = _get_parser(lang)
    if parser is None:
        return _whole_file_chunk(rel_path, lang, text, max_chunk_chars)

    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception as e:  # pragma: no cover
        log.debug("parse failed %s: %s", rel_path, e)
        return _whole_file_chunk(rel_path, lang, text, max_chunk_chars)

    matches = _walk_tree(tree.root_node, source, rules)
    if not matches:
        return _whole_file_chunk(rel_path, lang, text, max_chunk_chars)

    chunks: list[Chunk] = []
    for symbol, kind, node in matches:
        content = _safe_decode(source[node.start_byte : node.end_byte])
        if not content.strip():
            continue
        # Rust: drop `pub mod foo;` / `mod foo;` forward declarations — they
        # produce 1-line chunks that flood top-K with junk and never contain
        # semantically useful content. Inline `mod foo { ... }` bodies are
        # kept (they have braces). (MED-3 in v0.1.3 audit.)
        if lang == "rust" and node.type == "mod_item" and _RUST_MOD_DECL_RE.match(content.strip()):
            continue
        if len(content) > max_chunk_chars:
            # Split oversized code unit at line boundaries rather than dropping.
            chunks.extend(
                _split_oversized(
                    rel_path=rel_path,
                    lang=lang,
                    symbol=symbol,
                    kind=kind,
                    start_line=node.start_point[0] + 1,
                    content=content,
                    max_chunk_chars=max_chunk_chars,
                )
            )
            continue
        chunks.append(
            Chunk(
                path=rel_path,
                lang=lang,
                symbol=symbol,
                kind=kind,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                content=content,
                content_hash=_hash_content(content),
            )
        )
    if not chunks:
        return _whole_file_chunk(rel_path, lang, text, max_chunk_chars)
    return chunks


def chunk_text(
    rel_path: str,
    lang: str,
    text: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk a single text blob. Public entry for tests and direct use.

    `max_chunk_chars` caps each emitted chunk; oversized syntactic units are
    split at line boundaries (not dropped). Default sized for 2048-token
    Ollama embedding models.
    """
    if lang == "markdown":
        return _chunk_markdown(rel_path, text, max_chunk_chars)
    return _chunk_with_treesitter(rel_path, lang, text, max_chunk_chars)


def chunk_file(
    file_path: Path,
    repo_root: Path,
    max_bytes: int,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Chunk a single on-disk file. Returns [] on skip/error.

    v0.1.7 (MED audit): files that exceed ``max_bytes`` emit a WARNING log so
    operators don't have to deduce a silent skip from a missing chunk count.
    See ``index_repo`` which also counts these into ``chunker_skipped_files``.
    """
    try:
        size = file_path.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
        log.warning(
            "chunker skip: %s (size %d > max_file_bytes %d) — raise index.max_file_bytes "
            "in .codeindex/config.toml if this file should be indexed",
            file_path,
            size,
            max_bytes,
        )
        return []

    lang = detect_lang(file_path)
    if lang is None:
        return []

    try:
        raw = file_path.read_bytes()
    except OSError:
        return []
    text = _safe_decode(raw)
    if not text.strip():
        return []

    try:
        rel = str(file_path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        rel = str(file_path)
    return chunk_text(rel, lang, text, max_chunk_chars=max_chunk_chars)
