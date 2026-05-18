"""Structured logging via rich."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def get_console() -> Console:
    """Return shared stderr Console (so stdout stays clean for MCP/JSON output)."""
    return _console


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with Rich formatting.

    Honors CODE_INTEL_LOG env var if `level` is None. Default WARNING.
    """
    resolved = (level or os.environ.get("CODE_INTEL_LOG") or "WARNING").upper()
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
