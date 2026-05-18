"""Tree-sitter AST chunker. Extracts function/class/section chunks from source files."""

from __future__ import annotations

import hashlib
import importlib
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from code_intel._logging import get_logger

log = get_logger(__name__)

# Maximum chunk size in characters. Skip larger (likely generated code).
MAX_CHUNK_CHARS = 8000
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


def _whole_file_chunk(rel_path: str, lang: str, text: str) -> Chunk:
    lines = text.splitlines()
    head = "\n".join(lines[:WHOLE_FILE_FALLBACK_LINES])
    return Chunk(
        path=rel_path,
        lang=lang,
        symbol=Path(rel_path).name,
        kind="module",
        start_line=1,
        end_line=min(len(lines), WHOLE_FILE_FALLBACK_LINES),
        content=head,
        content_hash=_hash_content(head),
    )


def _chunk_markdown(rel_path: str, text: str) -> list[Chunk]:
    """Split markdown by H1/H2 headings."""
    lines = text.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    current_title = Path(rel_path).stem
    current_start = 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith(("# ", "## ")):
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
        if len(content) > MAX_CHUNK_CHARS:
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
        chunks.append(_whole_file_chunk(rel_path, "markdown", text))
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


def _chunk_with_treesitter(rel_path: str, lang: str, text: str) -> list[Chunk]:
    rules = LANG_NODE_RULES.get(lang)
    if rules is None:
        return [_whole_file_chunk(rel_path, lang, text)]

    parser = _get_parser(lang)
    if parser is None:
        return [_whole_file_chunk(rel_path, lang, text)]

    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception as e:  # pragma: no cover
        log.debug("parse failed %s: %s", rel_path, e)
        return [_whole_file_chunk(rel_path, lang, text)]

    matches = _walk_tree(tree.root_node, source, rules)
    if not matches:
        return [_whole_file_chunk(rel_path, lang, text)]

    chunks: list[Chunk] = []
    for symbol, kind, node in matches:
        content = _safe_decode(source[node.start_byte : node.end_byte])
        if not content.strip():
            continue
        if len(content) > MAX_CHUNK_CHARS:
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
        return [_whole_file_chunk(rel_path, lang, text)]
    return chunks


def chunk_text(rel_path: str, lang: str, text: str) -> list[Chunk]:
    """Chunk a single text blob. Public entry for tests and direct use."""
    if lang == "markdown":
        return _chunk_markdown(rel_path, text)
    return _chunk_with_treesitter(rel_path, lang, text)


def chunk_file(file_path: Path, repo_root: Path, max_bytes: int) -> list[Chunk]:
    """Chunk a single on-disk file. Returns [] on skip/error."""
    try:
        size = file_path.stat().st_size
    except OSError:
        return []
    if size > max_bytes:
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
    return chunk_text(rel, lang, text)
