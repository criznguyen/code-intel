"""Regression tests for the watcher-cosmetics mini-wave.

Two fixes, both post-v0.1.7 cosmetic / belt-and-suspenders:

* Fix A (Gap 6 LOW) — ``code_intel.watcher.main`` wraps ``run`` in a
  ``KeyboardInterrupt`` guard so SIGINT exits 0 with no Python traceback
  reaching stderr. ``asyncio.run`` would otherwise surface
  ``CancelledError`` → ``KeyboardInterrupt`` as a noisy stack trace.

* Fix B (belt-and-suspenders) — both ``code-intel-watcher@.service`` and
  ``code-intel-mcp@.service`` systemd templates set
  ``Environment="MALLOC_ARENA_MAX=2"`` explicitly. The watcher unit invokes
  ``python -m code_intel.watcher`` directly, bypassing the cli.py self-reexec
  bootstrap shipped in v0.1.7, so without this env line the arena cap would
  silently NOT apply under systemd.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates" / "systemd"


# ---------------------------------------------------------------------------
# Fix A — watcher SIGINT graceful exit
# ---------------------------------------------------------------------------


def test_watcher_main_handles_sigint_cleanly(tmp_path: Path) -> None:
    """Spawn ``python -m code_intel.watcher <target>``, send SIGINT, expect
    exit code 0 and NO ``Traceback`` keyword in combined stdout/stderr.

    The watcher only logs `watching <root>` after embedder + config bootstrap,
    so we poll stderr briefly to confirm the loop is alive before signalling.
    """
    target = tmp_path / "project"
    target.mkdir()
    # Minimal config so load_config() doesn't blow up looking for sources.
    (target / "README.md").write_text("# stub\n")

    env = os.environ.copy()
    # Force in-tree src/ on path so the test exercises THIS worktree.
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Skip arena cap to keep test fast (no extra fork+exec).
    env["CODE_INTEL_DISABLE_ARENA_CAP"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "code_intel.watcher", str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        # Wait until watcher logs "watching" (loop alive) or 8s timeout.
        deadline = time.monotonic() + 8.0
        started = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            # Non-blocking-ish read: check if process responding by sleeping.
            time.sleep(0.2)
            # Heuristic: after ~2s the asyncio loop should be in awatch.
            if time.monotonic() - (deadline - 8.0) > 2.5:
                started = True
                break

        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"watcher exited before SIGINT (rc={proc.returncode}); "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        assert started, "watcher did not appear to start within 2.5s"

        proc.send_signal(signal.SIGINT)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(f"watcher did not exit within 10s of SIGINT; stderr={stderr!r}")

        combined = (stdout or "") + (stderr or "")
        assert proc.returncode == 0, (
            f"expected exit 0 on SIGINT, got rc={proc.returncode}; "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
        assert "Traceback" not in combined, (
            f"SIGINT must not produce a Python traceback; got:\n{combined}"
        )
        assert "KeyboardInterrupt" not in combined, (
            f"SIGINT must not surface raw KeyboardInterrupt; got:\n{combined}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Fix B — systemd template carries MALLOC_ARENA_MAX=2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["code-intel-watcher@.service", "code-intel-mcp@.service"])
def test_systemd_template_has_arena_cap(unit: str) -> None:
    """The watcher unit MUST set MALLOC_ARENA_MAX=2 because it bypasses
    cli.py's self-reexec bootstrap. The mcp unit duplicates the env line as
    belt-and-suspenders (saves one fork+exec)."""
    path = TEMPLATES_DIR / unit
    assert path.exists(), f"systemd template missing: {path}"
    body = path.read_text()
    assert "MALLOC_ARENA_MAX=2" in body, (
        f"{unit} must set MALLOC_ARENA_MAX=2 in [Service]; got:\n{body}"
    )
    # Must live under [Service], not [Unit] / [Install].
    service_section = body.split("[Service]", 1)[1].split("[Install]", 1)[0]
    assert "MALLOC_ARENA_MAX=2" in service_section, (
        f"{unit}: MALLOC_ARENA_MAX=2 must be inside [Service] section"
    )
