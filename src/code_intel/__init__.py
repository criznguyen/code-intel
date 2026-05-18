"""code-intel: code intelligence MCP server for AI agents working on large repos."""

from __future__ import annotations

__version__ = "0.1.0"

from code_intel.config import Config, load_config

__all__ = ["Config", "__version__", "load_config"]
