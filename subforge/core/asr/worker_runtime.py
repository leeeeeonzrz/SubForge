"""Shared lifecycle helpers for isolated ASR worker processes."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def hidden_worker_process_kwargs() -> dict[str, Any]:
    """Return Windows process options that keep packaged workers invisible.

    ``CREATE_NO_WINDOW`` prevents a console from being allocated for console
    children. ``STARTF_USESHOWWINDOW``/``SW_HIDE`` also covers executable and
    bootloader combinations that briefly create a top-level window before the
    worker entry point takes control.
    """
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def atomic_json_write(path: Path, payload: object) -> None:
    """Publish a JSON worker message without exposing a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def stop_process(
    process: subprocess.Popen[Any],
    *,
    terminate_timeout: float = 5,
    kill_timeout: float = 10,
) -> None:
    """Stop a worker gracefully, escalating only when it does not exit."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=terminate_timeout)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=kill_timeout)


def log_tail(path: Path, limit: int = 4000) -> str:
    """Read the useful end of a worker log without failing error recovery."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
    except OSError:
        return ""
