"""Typer CLI surface for code-intel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from code_intel import __version__
from code_intel._logging import setup_logging
from code_intel.config import (
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
    console.print(
        f"[green]indexed[/] files={stats['files']} chunks={stats['chunks']} "
        f"embedded={stats['embedded']}{suffix}{prune_suffix}{cache_suffix}"
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
) -> None:
    """Quick semantic-search probe for debugging."""
    target_path = _resolve_target(target)
    cfg = load_config(target_path)
    from code_intel.search import quick_cli_search

    console.print(quick_cli_search(cfg, query, k=k))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
