"""Dependency / health checks. Prints PASS/WARN/FAIL with hints."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from code_intel.config import Config


@dataclass
class CheckResult:
    name: str
    level: str  # "PASS" | "WARN" | "FAIL"
    detail: str


def _exec_check(cmd: list[str]) -> tuple[bool, str]:
    if shutil.which(cmd[0]) is None:
        return False, f"{cmd[0]} not in PATH"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        first = (out.stdout or out.stderr).splitlines()
        return True, first[0] if first else "(no output)"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]} timed out"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _check_tool(name: str, cmd: list[str], hint: str, required: bool = True) -> CheckResult:
    ok, detail = _exec_check(cmd)
    if ok:
        return CheckResult(name, "PASS", detail)
    return CheckResult(name, "FAIL" if required else "WARN", f"{detail}. Hint: {hint}")


def _check_ollama_model(cfg: Config) -> CheckResult:
    if shutil.which("ollama") is None:
        return CheckResult(
            "ollama-model",
            "WARN",
            f"ollama not in PATH; install + `ollama pull {cfg.embedding.model}`",
        )
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10, check=False
        )
    except Exception as e:  # pragma: no cover
        return CheckResult("ollama-model", "WARN", str(e))
    if cfg.embedding.model.split(":")[0] in out.stdout:
        return CheckResult("ollama-model", "PASS", f"{cfg.embedding.model} available")
    return CheckResult(
        "ollama-model",
        "WARN",
        f"model '{cfg.embedding.model}' not pulled. Run `ollama pull {cfg.embedding.model}`",
    )


def _check_lancedb_writable(cfg: Config) -> CheckResult:
    p = cfg.lancedb_path
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".write-test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
        return CheckResult("lancedb-path", "PASS", f"writable at {p}")
    except Exception as e:
        return CheckResult("lancedb-path", "FAIL", f"{p}: {e}")


def _check_config(cfg: Config) -> CheckResult:
    if not cfg.config_path.exists():
        return CheckResult(
            "config",
            "WARN",
            f"no config at {cfg.config_path}; run `code-intel init` first",
        )
    return CheckResult("config", "PASS", str(cfg.config_path))


def run_doctor(target: Path) -> list[CheckResult]:
    from code_intel.config import load_config

    cfg = load_config(target)
    results: list[CheckResult] = []
    results.append(
        _check_tool("ripgrep", ["rg", "--version"], "https://github.com/BurntSushi/ripgrep")
    )
    results.append(
        _check_tool(
            "ast-grep",
            ["ast-grep", "--version"],
            "cargo install ast-grep / brew install ast-grep",
        )
    )
    results.append(_check_tool("fd", ["fd", "--version"], "brew/apt install fd-find"))
    results.append(_check_ollama_model(cfg))
    results.append(
        _check_tool(
            "docker",
            ["docker", "--version"],
            "needed only when [zoekt].enabled=true",
            required=False,
        )
    )
    results.append(
        _check_tool(
            "basedpyright",
            ["basedpyright", "--version"],
            "needed only when LSP support enabled (v0.2+)",
            required=False,
        )
    )
    results.append(_check_lancedb_writable(cfg))
    results.append(_check_config(cfg))
    return results


def format_results(results: list[CheckResult]) -> str:
    out_lines: list[str] = []
    width = max(len(r.name) for r in results)
    for r in results:
        out_lines.append(f"  [{r.level:<4}] {r.name.ljust(width)}  {r.detail}")
    return "\n".join(out_lines)


def has_failures(results: list[CheckResult]) -> bool:
    return any(r.level == "FAIL" for r in results)


def shell_path_hint() -> str:
    return f"PATH={os.environ.get('PATH', '')}"
