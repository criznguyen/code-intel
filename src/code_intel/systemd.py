"""Render + install systemd user units."""

from __future__ import annotations

import sys
from pathlib import Path

import tomli_w

from code_intel._logging import get_logger

log = get_logger(__name__)

USER_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
PROJECTS_DIR = Path.home() / ".config" / "code-intel" / "projects"

UNIT_NAMES = [
    "code-intel-mcp@.service",
    "code-intel-watcher@.service",
    "code-intel-zoekt@.service",
]


def _templates_dir() -> Path:
    """Locate the bundled templates dir.

    Strategy: look for an installed copy under the package's `_templates`
    (force-included by hatchling), then for a sibling `templates/` (dev mode).
    """
    here = Path(__file__).resolve().parent
    bundled = here / "_templates"
    if bundled.exists():
        return bundled
    dev = here.parent.parent / "templates"
    if dev.exists():
        return dev
    raise FileNotFoundError(f"Could not locate templates dir (looked at {bundled} and {dev})")


def _read_template(name: str) -> str:
    return (_templates_dir() / name).read_text(encoding="utf-8")


def write_project_manifest(instance: str, target: Path) -> Path:
    """Persist `instance -> target_path` so systemd units can resolve `%i`."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = PROJECTS_DIR / f"{instance}.toml"
    payload = {"instance": instance, "target": str(target.resolve())}
    with manifest.open("wb") as f:
        tomli_w.dump(payload, f)
    return manifest


def install_units() -> list[Path]:
    """Copy unit templates into ~/.config/systemd/user/. Returns written paths."""
    USER_UNIT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in UNIT_NAMES:
        src = f"systemd/{name}"
        content = _read_template(src).replace("{{ python_bin }}", sys.executable)
        dst = USER_UNIT_DIR / name
        dst.write_text(content, encoding="utf-8")
        written.append(dst)
    return written


def install_for_instance(instance: str, target: Path) -> dict[str, str]:
    """Install units + manifest for a given (instance, target). Returns summary."""
    units = install_units()
    manifest = write_project_manifest(instance, target)
    return {
        "manifest": str(manifest),
        "units": ", ".join(str(p) for p in units),
        "next": (
            f"systemctl --user daemon-reload && "
            f"systemctl --user enable --now code-intel-mcp@{instance}"
        ),
    }


def render_mcp_entry(target: Path, scope: str = "project") -> dict:
    """Return the MCP entry object for ~/.claude.json or .mcp.json.

    Shape:
        {
            "command": "code-intel",
            "args": ["serve", "--target", "<path>", "--stdio"],
            "env": {}
        }
    """
    return {
        "command": "code-intel",
        "args": ["serve", "--target", str(target.resolve()), "--stdio"],
        "env": {},
    }
