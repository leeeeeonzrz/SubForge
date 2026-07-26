import base64
import sys
import threading
import types

import launcher


def test_configure_frozen_standard_streams_replaces_missing_streams(monkeypatch, tmp_path):
    devnull = tmp_path / "null"
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "stdout", None)
    monkeypatch.setattr(launcher.sys, "stderr", None)
    monkeypatch.setattr(launcher.os, "devnull", str(devnull))

    launcher._configure_frozen_standard_streams()

    assert launcher.sys.stdout is not None
    assert launcher.sys.stderr is not None
    launcher.sys.stdout.write("stdout")
    launcher.sys.stderr.write("stderr")


def test_configure_runtime_paths_adds_source_backend(monkeypatch, tmp_path):
    launcher_file = tmp_path / "launcher.py"
    backend = tmp_path / "backend"
    backend.mkdir()
    launcher_file.touch()
    monkeypatch.setattr(launcher.sys, "frozen", False, raising=False)
    monkeypatch.setattr(launcher, "__file__", str(launcher_file))
    monkeypatch.setattr(launcher.sys, "path", [])

    launcher._configure_frozen_runtime_paths()

    assert launcher.sys.path == [str(backend)]


def test_configure_frozen_runtime_paths_adds_macos_bundle_roots(monkeypatch, tmp_path):
    contents = tmp_path / "SubForge.app" / "Contents"
    executable = contents / "MacOS" / "SubForge"
    resources = contents / "Resources"
    frameworks = contents / "Frameworks"
    runtime = frameworks
    for path in (executable.parent, resources, frameworks):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(runtime), raising=False)
    monkeypatch.setattr(launcher.sys, "executable", str(executable))
    original_path = list(launcher.sys.path)
    monkeypatch.setattr(launcher.sys, "path", original_path.copy())

    launcher._configure_frozen_runtime_paths()

    assert str(resources) in launcher.sys.path
    assert str(frameworks) in launcher.sys.path


def test_cleanup_desktop_session_requests_shutdown_and_removes_uploads(monkeypatch):
    calls = []

    class FakeServer:
        should_exit = False

    server = FakeServer()
    monkeypatch.setattr("app.api.files.cleanup_session_uploads", lambda: calls.append("cleanup"))

    launcher._cleanup_desktop_session(server)

    assert server.should_exit is True
    assert calls == ["cleanup"]


def test_cleanup_desktop_session_can_leave_active_uploads_in_place(monkeypatch):
    calls = []

    class FakeServer:
        should_exit = False

    server = FakeServer()
    monkeypatch.setattr("app.api.files.cleanup_session_uploads", lambda: calls.append("cleanup"))

    launcher._cleanup_desktop_session(server, cleanup_uploads=False)

    assert server.should_exit is True
    assert calls == []


def test_desktop_shutdown_runs_off_ui_thread_and_is_idempotent(monkeypatch):
    cleanup_calls = []
    exit_codes = []
    finished = threading.Event()

    class FakeServerThread:
        def join(self, timeout):
            assert timeout == 1.5

        @staticmethod
        def is_alive():
            return True

    monkeypatch.setattr(
        launcher,
        "_cleanup_desktop_session",
        lambda server=None, **kwargs: cleanup_calls.append((server, kwargs)),
    )
    started = threading.Event()

    def _hard_exit(code):
        exit_codes.append(code)
        finished.set()

    server = object()
    assert launcher._begin_desktop_shutdown(
        server,
        FakeServerThread(),
        started,
        hard_exit=_hard_exit,
    )
    assert not launcher._begin_desktop_shutdown(
        server,
        FakeServerThread(),
        started,
        hard_exit=_hard_exit,
    )
    assert finished.wait(timeout=1)
    assert cleanup_calls == [(server, {"cleanup_uploads": False})]
    assert exit_codes == [0]


def test_start_server_disables_console_logging(monkeypatch):
    captured = {}

    class FakeConfig:
        def __init__(self, app, **kwargs):
            captured["app"] = app
            captured.update(kwargs)

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            captured["ran"] = True

    fake_uvicorn = type("FakeUvicorn", (), {"Config": FakeConfig, "Server": FakeServer})
    monkeypatch.setitem(launcher.sys.modules, "uvicorn", fake_uvicorn)

    errors = []
    launcher.start_server(8765, errors)

    assert errors == []
    assert captured["app"] == "app.main:app"
    assert captured["log_config"] is None
    assert captured["access_log"] is False
    assert captured["ran"] is True


def test_start_server_preserves_exception_type(monkeypatch, tmp_path):
    class FailingConfig:
        def __init__(self, *args, **kwargs):
            raise ValueError("bad configuration")

    fake_uvicorn = type("FakeUvicorn", (), {"Config": FailingConfig, "Server": object})
    monkeypatch.setitem(launcher.sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(tmp_path))

    errors = []
    launcher.start_server(8765, errors)

    assert errors == ["ValueError: bad configuration"]
    startup_log = tmp_path / "SubForge" / "startup-error.log"
    assert "ValueError: bad configuration" in startup_log.read_text(encoding="utf-8")


def test_native_file_dialog_grants_selected_path(monkeypatch, tmp_path):
    selected = tmp_path / "external" / "video.mp4"
    selected.parent.mkdir()
    selected.touch()
    granted = []

    class FakeWindow:
        @staticmethod
        def create_file_dialog(*_args, **_kwargs):
            return (str(selected),)

    fake_webview = types.SimpleNamespace(
        windows=[FakeWindow()],
        OPEN_DIALOG="open",
        SAVE_DIALOG="save",
    )
    monkeypatch.setitem(launcher.sys.modules, "webview", fake_webview)
    monkeypatch.setattr("app.security.grant_path", lambda path: granted.append(path) or selected)

    result = launcher.Api().open_file("media")

    assert result == {"ok": True, "path": str(selected)}
    assert granted == [str(selected)]


def test_save_dialog_is_dispatched_to_windows_ui_thread(monkeypatch):
    calls = []

    class NativeWindow:
        InvokeRequired = True

        def Invoke(self, callback):
            calls.append("invoke")
            callback()

    native_window = NativeWindow()
    fake_gui = type(
        "FakeGui",
        (),
        {
            "BrowserView": type("BrowserView", (), {"instances": {"window-id": native_window}}),
            "Func": type("Func", (), {"__class_getitem__": classmethod(lambda cls, item: lambda fn: fn)}),
            "Type": object,
        },
    )

    class FakeWindow:
        uid = "window-id"
        gui = fake_gui

        def create_file_dialog(self, *args, **kwargs):
            calls.append((args, kwargs))
            return (r"C:\\exports\\captions.srt",)

    fake_webview = type(
        "FakeWebview",
        (),
        {"FileDialog": type("FileDialog", (), {"SAVE": 20})},
    )
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(launcher.sys, "platform", "win32")

    result = launcher.Api._show_save_dialog(FakeWindow(), "captions.srt", r"C:\\exports")

    assert result == (r"C:\\exports\\captions.srt",)
    assert calls[0] == "invoke"
    assert calls[1][1]["directory"] == r"C:\\exports"


def test_save_file_reports_cancelled_without_writing(monkeypatch, tmp_path):
    fake_webview = type("FakeWebview", (), {"windows": [object()]})
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(launcher.Api, "_save_dialog_directory", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(launcher.Api, "_show_save_dialog", staticmethod(lambda *args: None))

    result = launcher.Api().save_file(base64.b64encode(b"subtitle").decode(), "test.srt")

    assert result == {"ok": False, "cancelled": True}
    assert list(tmp_path.iterdir()) == []


def test_save_file_writes_selected_path_atomically(monkeypatch, tmp_path):
    output = tmp_path / "translated.srt"
    fake_webview = type("FakeWebview", (), {"windows": [object()]})
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(launcher.Api, "_save_dialog_directory", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(
        launcher.Api,
        "_show_save_dialog",
        staticmethod(lambda *args: (str(output),)),
    )

    result = launcher.Api().save_file(
        base64.b64encode("译文".encode()).decode(),
        "translated.srt",
    )

    assert result == {"ok": True, "path": str(output)}
    assert output.read_text(encoding="utf-8") == "译文"
