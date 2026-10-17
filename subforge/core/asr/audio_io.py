"""Resource-safe pydub file helpers."""

from __future__ import annotations

import wave
from os import PathLike
from pathlib import Path
from typing import Any

from pydub import AudioSegment


def load_audio_file(path: str | PathLike[str]) -> Any:
    """Decode an audio file while deterministically closing its input handle."""
    with open(path, "rb") as source:
        # pydub cannot infer an extension from an open file handle. Preserve
        # its native reader only for confirmed PCM WAV: its own WAV parser
        # can mistake WAVE_FORMAT_EXTENSIBLE floating-point samples for PCM.
        if Path(path).suffix.lower() in {".wav", ".wave"}:
            is_pcm = False
            try:
                with wave.open(source, "rb"):
                    is_pcm = True
            except (wave.Error, EOFError):
                pass
            source.seek(0)
            if is_pcm:
                return AudioSegment.from_file(source, format="wav")
        return AudioSegment.from_file(source)


def export_audio_file(
    audio: Any,
    path: str | PathLike[str],
    *,
    format: str,
    **kwargs: Any,
) -> None:
    """Export to a path and close the file object returned by pydub."""
    output: Any = audio.export(str(path), format=format, **kwargs)
    close = getattr(output, "close", None)
    if callable(close):
        close()
