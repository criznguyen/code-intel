"""Memory regression bench: run N semantic searches and print RSS delta.

Use after material changes to embedder, store, or search hot path. Reference
point from v0.1.3 audit (INFO-10): 50 queries → ~650MB RSS due to per-call
LanceDB connection. v0.1.4 LOW-9 cache should drop that materially; this
script makes that observable without ceremony.

Usage::

    uv run python scripts/bench_memory.py \
        --target /path/to/repo \
        --query "calculate fee" \
        --n 50

Prints two RSS lines (before and after the loop) plus elapsed wall time.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

# Make `code_intel` importable when this script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# v0.1.7: import cli FIRST so the MALLOC_ARENA_MAX re-exec bootstrap fires
# before any LanceDB / Arrow alloc. Without this, the bench under-reports the
# real production plateau because uv-run wasn't routed through the CLI shim.
# Importing cli here is cheap (typer is already a dep) and idempotent: when
# `MALLOC_ARENA_MAX` is already set by the operator, no re-exec happens.
import code_intel.cli  # noqa: F401, E402


def _rss_mb() -> float:
    """Process-only RSS in MB. ru_maxrss is in KB on Linux, bytes on macOS."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return r / (1024 * 1024)
    return r / 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="Indexed repo root")
    parser.add_argument("--query", type=str, default="calculate fee", help="Query string")
    parser.add_argument("--n", type=int, default=50, help="Number of searches")
    parser.add_argument("--k", type=int, default=10, help="Top-K per query")
    args = parser.parse_args()

    from code_intel.config import load_config
    from code_intel.search import semantic_search

    cfg = load_config(args.target)

    # Warm: 1 throwaway to amortize first-connect cost.
    semantic_search(cfg, args.query, k=args.k)

    before = _rss_mb()
    t0 = time.perf_counter()
    for _ in range(args.n):
        semantic_search(cfg, args.query, k=args.k)
    elapsed = time.perf_counter() - t0
    after = _rss_mb()

    print(f"queries     : {args.n}")
    print(f"elapsed     : {elapsed:.2f}s ({elapsed / args.n * 1000:.1f}ms / query)")
    print(f"rss before  : {before:.1f} MB")
    print(f"rss after   : {after:.1f} MB")
    print(f"rss delta   : {after - before:+.1f} MB")
    print(f"pid         : {os.getpid()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
