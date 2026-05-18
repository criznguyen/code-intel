"""Typer CLI surface for code-intel."""

from __future__ import annotations

# v0.1.7 NEW-1 perf: cap glibc's per-thread malloc arenas BEFORE any native
# extension (httpx, lancedb, pyarrow) gets a chance to allocate. The default
# `MALLOC_ARENA_MAX = 8 * ncores` causes severe allocator fragmentation on
# repeated LanceDB queries: 50 semantic_search calls on a 20 MB index pushed
# RssAnon from 71 MB → 617 MB on a 16-core box. Capping arenas at 2 traps
# the working set at ~220 MB plateau (62 % cut) with no measured latency
# regression. glibc reads MALLOC_ARENA_MAX once at process start, so we must
# re-exec ourselves if it wasn't set in the environment we inherited.
import os as _os
import sys as _sys

_ARENA_SENTINEL = "_CODE_INTEL_ARENA_BOOTSTRAPPED"


def _should_arena_bootstrap() -> bool:
    """Re-exec to inject MALLOC_ARENA_MAX only for top-level CLI invocations.

    We refuse to re-exec when we look like we're being imported as a library
    (interactive ``python``, ``python -c "..."``, pytest, IPython, etc.) because
    re-execing those would either lose user state or strip the inline code,
    and re-execing pytest mid-test-collection breaks its plugin state.

    Detection: argv[0] basename must look like a code-intel entry point —
    either an installed ``code-intel`` shim, the ``__main__`` module form, or
    the bench script. Anything else (``pytest``, ``ipython``, ``-c``, etc.)
    skips the cap so other Python workloads aren't disturbed.
    """
    if not _sys.platform.startswith("linux"):
        return False
    if _os.environ.get(_ARENA_SENTINEL) == "1":
        return False
    if _os.environ.get("CODE_INTEL_DISABLE_ARENA_CAP") == "1":
        return False
    argv0 = _sys.argv[0] if _sys.argv else ""
    if not argv0 or argv0.startswith("-"):
        return False
    try:
        if not _os.path.isfile(argv0):
            return False
    except OSError:
        return False
    basename = _os.path.basename(argv0)
    # Whitelist: known code-intel entry shapes only. ``__main__.py`` covers
    # ``python -m code_intel``; ``code-intel`` covers the installed uv-tool
    # shim; ``bench_memory.py`` covers the repo bench. Anything else (pytest,
    # ipython, gunicorn, …) silently skips the cap.
    allowed = {"__main__.py", "code-intel", "bench_memory.py"}
    return basename in allowed


if _should_arena_bootstrap():
    # v0.1.7 NEW-1 perf: cap glibc's per-thread malloc arenas BEFORE any native
    # extension (httpx, lancedb, pyarrow) gets a chance to allocate. The default
    # `MALLOC_ARENA_MAX = 8 * ncores` causes severe allocator fragmentation on
    # repeated LanceDB queries: 50 semantic_search calls on a 20 MB index pushed
    # RssAnon from 71 MB → 617 MB on a 16-core box. Capping arenas at 2 traps
    # the working set at ~220 MB plateau (62 % cut) with no measured latency
    # regression. glibc reads MALLOC_ARENA_MAX once at process start, so we
    # re-exec ourselves with the same argv to inject it.
    if "MALLOC_ARENA_MAX" not in _os.environ:
        _os.environ["MALLOC_ARENA_MAX"] = "2"
    _os.environ[_ARENA_SENTINEL] = "1"
    _os.execvpe(_sys.executable, [_sys.executable, *_sys.argv], _os.environ)

# Imports below intentionally trail the arena bootstrap block. Moving them
# above would defeat the cap because lancedb / pyarrow alloc on import.
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from code_intel import __version__  # noqa: E402
from code_intel._logging import setup_logging  # noqa: E402
from code_intel.config import (  # noqa: E402
    CODEINDEX_DIRNAME,
    CONFIG_FILENAME,
    default_config,
    load_config,
    save_config,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="code-intel: code intelligence MCP server for AI agents.",
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"code-intel {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
) -> None:
    setup_logging("INFO" if verbose else None)


def _resolve_target(target: Path | None) -> Path:
    return (target or Path.cwd()).resolve()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
@app.command()
def init(
    target: Path = typer.Option(None, "--target", help="Repo to bootstrap (default: cwd)."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config."),
) -> None:
    """Bootstrap target repo with .codeindex/ and a default config."""
    target_path = _resolve_target(target)
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    codeindex = target_path / CODEINDEX_DIRNAME
    cfg_path = codeindex / CONFIG_FILENAME

    if cfg_path.exists() and not force:
        err_console.print(f"[yellow]config already exists at[/] {cfg_path}")
        err_console.print("Use --force to overwrite.")
        raise typer.Exit(code=1)

    cfg = default_config(project_name=target_path.name)
    cfg._target = target_path
    save_config(cfg, target=target_path)

    # Sub-dirs.
    (codeindex / "digests").mkdir(parents=True, exist_ok=True)
    (codeindex / "lancedb").mkdir(parents=True, exist_ok=True)

    # .gitignore snippet (append-if-missing).
    gitignore = target_path / ".gitignore"
    snippet = "\n# code-intel artefacts\n.codeindex/\n"
    if gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8", errors="replace")
        if ".codeindex/" not in existing:
            gitignore.write_text(existing.rstrip() + snippet, encoding="utf-8")
    else:
        gitignore.write_text(snippet.lstrip(), encoding="utf-8")

    console.print(f"[green]initialized[/] code-intel at {target_path}")
    console.print(f"  config: {cfg_path}")
    console.print(f"  embedding: {cfg.embedding.provider}/{cfg.embedding.model}")
    console.print("  next: run [cyan]code-intel doctor[/] then [cyan]code-intel index[/]")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
@app.command()
def index(
    target: Path = typer.Option(None, "--target"),
    full: bool = typer.Option(False, "--full", help="Reindex everything (default)."),
    since: str = typer.Option(
        None, "--since", help="Reindex files changed since a git ref (e.g. HEAD~5)."
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help="Remove DB rows for files that no longer match include_globs / exist on disk.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Bypass content-hash cache during --since (use after changing embedding model/dim).",
    ),
) -> None:
    """Run the chunk + embed pipeline."""
    target_path = _resolve_target(target)
    cfg = load_config(target_path)

    if full and since:
        err_console.print("[red]--full and --since are mutually exclusive[/]")
        raise typer.Exit(code=2)

    # v0.1.6 NEW-1: `index` is the only long-running CLI command. Default
    # WARNING level made foreground runs look frozen for 10-30 min (no
    # per-batch progress). Auto-elevate to INFO so users get periodic
    # "discovered N files / produced M chunks / embedded B/T batch" lines
    # without needing to remember `-v`. `--verbose` (and CODE_INTEL_LOG=DEBUG)
    # still upgrades further; explicit CODE_INTEL_LOG=WARNING wins.
    import logging as _logging
    import os as _os
    if not _os.environ.get("CODE_INTEL_LOG"):
        root_lvl = _logging.getLogger().getEffectiveLevel()
        if root_lvl > _logging.INFO:
            setup_logging("INFO")

    try:
        from code_intel.indexer import index_repo, prune_orphans

        pruned = 0
        if prune:
            pruned = prune_orphans(cfg)
        stats = index_repo(cfg, since=since if since else None, force=force)
    except Exception as e:
        err_console.print(f"[red]index failed:[/] {e}")
        raise typer.Exit(code=1) from e
    skipped = stats.get("skipped", 0)
    suffix = f" skipped={skipped} (see warnings above)" if skipped else ""
    prune_suffix = f" pruned={pruned}" if prune else ""
    cache_hits = stats.get("cache_hits", 0)
    cache_suffix = f" cache_hits={cache_hits}" if cache_hits else ""
    # v0.1.7 MED: surface files that chunker dropped for exceeding max_file_bytes
    # so operators don't have to deduce a silent skip from a missing chunk count.
    chunker_skipped = stats.get("chunker_skipped_files", 0)
    chunker_suffix = (
        f" chunker_skipped_files={chunker_skipped} (over max_file_bytes; see warnings)"
        if chunker_skipped
        else ""
    )
    console.print(
        f"[green]indexed[/] files={stats['files']} chunks={stats['chunks']} "
        f"embedded={stats['embedded']}{suffix}{prune_suffix}{cache_suffix}{chunker_suffix}"
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    target: Path = typer.Option(None, "--target"),
    socket: Path = typer.Option(None, "--socket", help="Unused in v0.1 (stdio only)."),
    stdio: bool = typer.Option(True, "--stdio/--no-stdio", help="Use stdio transport."),
) -> None:
    """Run the MCP server (stdio transport by default)."""
    target_path = _resolve_target(target)
    if socket is not None:
        err_console.print(
            "[yellow]--socket transport is documented but not implemented in v0.1; "
            "falling back to stdio.[/]"
        )
    if not stdio:
        err_console.print("[red]Only stdio transport is supported in v0.1.[/]")
        raise typer.Exit(code=2)

    try:
        from code_intel.mcp_server import run_stdio

        run_stdio(target_path)
    except Exception as e:
        err_console.print(f"[red]serve failed:[/] {e}")
        raise typer.Exit(code=1) from e


# ---------------------------------------------------------------------------
# install-services
# ---------------------------------------------------------------------------
@app.command("install-services")
def install_services(
    instance: str = typer.Option(..., "--instance", help="Instance name, e.g. 'solanabot'."),
    target: Path = typer.Option(None, "--target"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing unit files (recovers from stale templates)."
    ),
) -> None:
    """Render + install systemd user units for a given (instance, target)."""
    target_path = _resolve_target(target)
    from code_intel.systemd import install_for_instance

    info = install_for_instance(instance, target_path, force=force)
    console.print(f"[green]installed services[/] for instance '{instance}'")
    console.print(f"  manifest: {info['manifest']}")
    console.print(f"  units:    {info['units']}")
    console.print(f"  next:     {info['next']}")


# ---------------------------------------------------------------------------
# mcp-config
# ---------------------------------------------------------------------------
@app.command("mcp-config")
def mcp_config(
    target: Path = typer.Option(None, "--target"),
    scope: str = typer.Option("project", "--scope", help="'project' or 'user'."),
) -> None:
    """Print the JSON entry to drop into ~/.claude.json or .mcp.json.

    Project-scoped entries (`.mcp.json` in the repo) are already namespaced
    by repo path, so the key is just `"code-intel"`. User-scoped entries
    (`~/.claude.json`) are shared across all projects, so they're keyed
    `"code-intel-<project>"` to avoid collisions.
    """
    target_path = _resolve_target(target)
    from code_intel.systemd import render_mcp_entry

    entry = render_mcp_entry(target_path, scope=scope)
    if scope == "project":
        key = "code-intel"
    elif scope == "user":
        key = f"code-intel-{target_path.name}"
    else:
        err_console.print(f"[red]invalid --scope: {scope!r} (use 'project' or 'user')[/]")
        raise typer.Exit(code=2)
    payload = {"mcpServers": {key: entry}}
    console.print_json(json.dumps(payload))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
@app.command()
def doctor(
    target: Path = typer.Option(None, "--target"),
) -> None:
    """Health-check binaries, models, and config."""
    target_path = _resolve_target(target)
    from code_intel.doctor import format_results, has_failures, run_doctor

    results = run_doctor(target_path)
    console.print(f"[bold]code-intel doctor[/]  ({target_path})")
    console.print(format_results(results))
    if has_failures(results):
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# search (hidden, dev)
# ---------------------------------------------------------------------------
@app.command(hidden=True)
def search(
    query: str = typer.Argument(...),
    target: Path = typer.Option(None, "--target"),
    k: int = typer.Option(5, "--k"),
    rerank: bool = typer.Option(
        True,
        "--rerank/--no-rerank",
        help="Apply heuristic reranker over LanceDB candidates (v0.1.5+). "
        "Use --no-rerank to inspect raw L2 nearest-neighbor order, e.g. for "
        "embedding-model A/B benchmarks.",
    ),
) -> None:
    """Quick semantic-search probe for debugging."""
    target_path = _resolve_target(target)
    cfg = load_config(target_path)
    from code_intel.search import quick_cli_search

    console.print(quick_cli_search(cfg, query, k=k, rerank=rerank))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
