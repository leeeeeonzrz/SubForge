from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

from subforge.core.asr import worker_runtime
from subforge.core.asr.worker_runtime import atomic_json_write, log_tail, stop_process


def test_windows_worker_process_is_created_hidden(monkeypatch) -> None:
    class FakeStartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = None

    monkeypatch.setattr(worker_runtime.os, "name", "nt")
    monkeypatch.setattr(
        worker_runtime.subprocess,
        "STARTUPINFO",
        FakeStartupInfo,
        raising=False,
    )
    monkeypatch.setattr(worker_runtime.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(worker_runtime.subprocess, "STARTF_USESHOWWINDOW", 0x00000001, raising=False)
    monkeypatch.setattr(worker_runtime.subprocess, "SW_HIDE", 0, raising=False)

    kwargs = worker_runtime.hidden_worker_process_kwargs()

    assert kwargs["creationflags"] == 0x08000000
    assert kwargs["startupinfo"].dwFlags & 0x00000001
    assert kwargs["startupinfo"].wShowWindow == 0


def test_atomic_json_write_replaces_target_without_leaving_temporary_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"

    atomic_json_write(target, {"text": "中文", "ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "text": "中文",
        "ok": True,
    }
    assert not target.with_suffix(".json.tmp").exists()


def test_stop_process_escalates_when_worker_ignores_terminate() -> None:
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 5), 0]

    stop_process(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


def test_stop_process_does_nothing_for_finished_worker() -> None:
    process = Mock()
    process.poll.return_value = 0

    stop_process(process)

    process.terminate.assert_not_called()


def test_log_tail_handles_missing_file_and_limits_output(tmp_path: Path) -> None:
    path = tmp_path / "worker.log"
    assert log_tail(path) == ""

    path.write_text("abcdef", encoding="utf-8")
    assert log_tail(path, 3) == "def"
