# code-intel

A code-intelligence stack that gives AI agents fast, scoped access to large
codebases via the Model Context Protocol (MCP). Designed for solo developers
who reuse the same indexing pipeline across many repos.

## What it is

`code-intel` is a pip-installable Python package that ships:

- A **Typer CLI** (`code-intel`) for bootstrap, indexing, serving, and health
  checks.
- A **tree-sitter chunker** that extracts function/class/section-level chunks
  from Rust, Python, Go, TypeScript/JS, Java, Kotlin, and Markdown.
- An **embedder plugin** with a default `OllamaProvider` — embeddings live in
  a per-repo LanceDB store under `.codeindex/`. code-intel core is local-first
  by design; paid-API providers can be added via external plugin packages
  that implement the `EmbeddingProvider` Protocol.
- An **MCP server** (FastMCP, stdio transport) that Claude Code, Cursor, or any
  MCP-aware agent can attach to. Exposes: `search_lexical` (ripgrep),
  `semantic_search`, `structural` (ast-grep), `get_digest`, `list_modules`.
- **systemd user units** (instanced — `code-intel-mcp@<instance>`) so multiple
  repos can run side-by-side.

## Why

Existing options either lock you into a vendor (Cursor's hidden RAG), require a
heavy on-prem deployment (Sourcegraph + Zoekt), or skip vector recall entirely
(Aider's repo-map). `code-intel` is the small middle ground: a local CLI, a
local LanceDB index, a local Ollama embedder, and an MCP server your agent
already knows how to talk to. Zoekt and LSP support are documented stubs for
v0.2; ripgrep + tree-sitter cover v0.1.

## Quickstart

```bash
# 1. install
pipx install code-intel        # or: uv tool install code-intel

# 2. bootstrap a repo (creates .codeindex/ and a default config)
cd ~/myrepo
code-intel init

# 3. check binaries are present (rg, ast-grep, fd, ollama)
code-intel doctor

# 4. pull the default embedding model
ollama pull embeddinggemma

# 5. index the repo (chunk + embed -> LanceDB)
code-intel index

# 6. drop the MCP entry into ~/.claude.json and use the server from Claude Code
code-intel mcp-config --target . >> /tmp/mcp-snippet.json
```

## Architecture

```
+----------------+      +----------------+      +----------------+
|  CLI (Typer)   |----->|   Indexer      |----->|   LanceDB      |
+----------------+      | chunk + embed  |      |  .codeindex/   |
        |               +----------------+      +----------------+
        v                                              ^
+----------------+      +----------------+              |
| Claude Code /  |<-----|  FastMCP       |--------------+
| any MCP client | stdio|  mcp_server.py |    semantic_search
+----------------+      +----------------+    search_lexical (rg)
                                              structural    (ast-grep)
```

## CLI reference

| Command | Purpose |
|---|---|
| `code-intel init [--target PATH] [--force]` | Create `.codeindex/`, write default config, append `.gitignore`. |
| `code-intel index [--target PATH] [--full \| --since GIT_REF]` | Chunk + embed pipeline. `--since` reindexes only files changed since a git ref. |
| `code-intel serve [--target PATH] [--stdio]` | Run the MCP server (stdio). |
| `code-intel install-services --instance NAME [--target PATH]` | Render + install systemd user units; write a manifest at `~/.config/code-intel/projects/<instance>.toml`. |
| `code-intel mcp-config [--target PATH] [--scope project\|user]` | Print the MCP entry JSON for `~/.claude.json` or `.mcp.json`. |
| `code-intel doctor [--target PATH]` | Health-check binaries (`rg`, `ast-grep`, `fd`, `ollama`, `docker`, `basedpyright`), model availability, LanceDB writability. |
| `code-intel --version` | Show version. |

## Config reference (`.codeindex/config.toml`)

| Key | Default | Description |
|---|---|---|
| `project.name` | `<repo-basename>` | Display name (used in MCP server identifier). |
| `project.root` | `.` | Root, relative to the file. |
| `index.include_globs` | `**/*.{rs,py,go,ts,tsx,js,jsx,java,kt,md}` | Files to chunk. |
| `index.exclude_globs` | `target/`, `node_modules/`, `.venv/`, etc. | Filter applied after includes. |
| `index.max_file_bytes` | `1_000_000` | Skip files larger than this. |
| `embedding.provider` | `ollama` | Only `ollama` is shipped in core. External plugin packages can register more. |
| `embedding.model` | `embeddinggemma` | Model name as pulled into Ollama. |
| `embedding.endpoint` | `http://localhost:11434` | Ollama HTTP endpoint. |
| `embedding.batch_size` | `32` | Texts per HTTP round-trip. |
| `embedding.dim` | `768` | Vector dimension (match the model). |
| `lancedb.path` | `.codeindex/lancedb` | Vector store location (relative to target). |
| `lancedb.table` | `chunks` | Table name. |
| `mcp.transport` | `stdio` | Only `stdio` works in v0.1. |
| `zoekt.enabled` | `false` | Reserved for v0.2 (lexical via Zoekt). |

## Embedding model options

`code-intel` is provider-agnostic; for v0.1 the default is **Ollama** because
it's free, local, and works offline. Pick a model:

| Model | Dim | Notes |
|---|---|---|
| `embeddinggemma` (default) | 768 | Google EmbeddingGemma 308M; small, fast, multilingual. |
| `nomic-embed-text` | 768 | Good general-purpose; mature. |
| `mxbai-embed-large` | 1024 | Higher quality, ~3x slower; set `embedding.dim = 1024`. |

To change: edit `[embedding]` in `.codeindex/config.toml`, `ollama pull <model>`,
re-run `code-intel index`.

## Extending with external providers

code-intel core ships exactly one embedding provider: `OllamaProvider`. This
is deliberate — the project is local-first and we don't want to bundle clients
for paid APIs in the core package.

If you want Voyage, OpenAI, Cohere, or any other embedder, implement the
public `EmbeddingProvider` Protocol in your own package and register it before
calling `get_provider`:

```python
from code_intel.embedder import EmbeddingProvider, _REGISTRY

class MyProvider:
    name = "myprovider"
    dim = 1024

    def __init__(self, cfg): ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

_REGISTRY["myprovider"] = MyProvider
```

A future minor release may formalise this as a `code_intel.providers` entry
point group; until then the registry dict is the extension surface.

## Roadmap

- **v0.2** — Zoekt-backed lexical search (Docker), LSP integration via
  `basedpyright` / `rust-analyzer` (`go_to_definition`, `find_references`).
- **v0.3** — Automatic `digest.md` generation per top-level module via the
  Claude API; Tree-sitter coverage for C/C++/Swift/PHP.

## License

Apache-2.0. See `LICENSE`.
