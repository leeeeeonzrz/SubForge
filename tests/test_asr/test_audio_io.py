import builtins
import shutil
import struct
import subprocess
import wave

import pytest
from pydub import audio_segment, utils

from subforge.core.asr import audio_io


def _write_pcm_wav(path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 16_000)


@pytest.mark.parametrize("suffix", [".wav", ".WAV", ".wave", ".WaVe"])
@pytest.mark.parametrize("as_string", [False, True])
def test_pcm_wav_reads_without_external_process_and_closes_input(
    monkeypatch, tmp_path, suffix, as_string
):
    path = tmp_path / f"音频{suffix}"
    _write_pcm_wav(path)
    handles = []

    def track_open(*args, **kwargs):
        handle = builtins.open(*args, **kwargs)
        handles.append(handle)
        return handle

    def unexpected_process(*_args, **_kwargs):
        raise AssertionError("PCM WAV loading must not start ffprobe or ffmpeg")

    monkeypatch.setattr(audio_io, "open", track_open, raising=False)
    monkeypatch.setattr(audio_segment.subprocess, "Popen", unexpected_process)
    monkeypatch.setattr(utils, "Popen", unexpected_process)

    audio = audio_io.load_audio_file(str(path) if as_string else path)

    assert len(audio) == 1000
    assert audio.frame_rate == 16_000
    assert audio.channels == 1
    assert audio.sample_width == 2
    assert len(handles) == 1
    assert handles[0].closed
    path.unlink()  # Also catches an open Windows file handle.


@pytest.mark.parametrize("suffix", [".wav", ".mp3"])
def test_load_closes_input_when_decode_fails(monkeypatch, tmp_path, suffix):
    path = tmp_path / f"invalid{suffix}"
    path.write_bytes(b"invalid audio")
    handles = []

    def fail_decode(source, **kwargs):
        handles.append(source)
        assert not source.closed
        raise ValueError("invalid audio")

    monkeypatch.setattr(audio_io.AudioSegment, "from_file", fail_decode)

    with pytest.raises(ValueError, match="invalid audio"):
        audio_io.load_audio_file(path)

    assert len(handles) == 1
    assert handles[0].closed
    path.unlink()


def test_pcm_probe_does_not_hide_io_errors(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    _write_pcm_wav(path)

    def fail_read(*_args, **_kwargs):
        raise OSError("audio read failed")

    def unexpected_decode(*_args, **_kwargs):
        raise AssertionError("An I/O failure must not fall back to another decoder")

    monkeypatch.setattr(audio_io.wave, "open", fail_read)
    monkeypatch.setattr(audio_io.AudioSegment, "from_file", unexpected_decode)

    with pytest.raises(OSError, match="audio read failed"):
        audio_io.load_audio_file(path)

    path.unlink()


@pytest.mark.parametrize("extensible", [False, True])
def test_float_wav_preserves_amplitude_through_external_decoder(tmp_path, extensible):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required for floating-point WAV decoding")
    values = [0.0, 0.001, -0.001] * 16_000
    samples = struct.pack(f"<{len(values)}f", *values)
    if extensible:
        float_subtype = bytes.fromhex("0300000000001000800000aa00389b71")
        fmt = struct.pack("<HHIIHHHHI", 65534, 1, 16_000, 64_000, 4, 32, 22, 32, 4)
        fmt += float_subtype
    else:
        fmt = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    content = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    content += b"data" + struct.pack("<I", len(samples)) + samples
    path = tmp_path / "floating-point.wav"
    path.write_bytes(b"RIFF" + struct.pack("<I", len(content)) + content)

    audio = audio_io.load_audio_file(path)

    scale = 1 << (audio.sample_width * 8 - 1)
    normalized = [sample / scale for sample in audio.get_array_of_samples()[:3]]
    assert normalized == pytest.approx([0.0, 0.001, -0.001], abs=0.000001)
    assert len(audio) == 3000
    path.unlink()


@pytest.mark.parametrize("suffix", [".mp3", ".audio"])
def test_non_wav_keeps_external_format_detection(monkeypatch, tmp_path, suffix):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required for the compressed-audio check")
    source_path = tmp_path / "source.wav"
    _write_pcm_wav(source_path)
    path = tmp_path / f"compressed{suffix}"
    subprocess.run(
        [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path), "-codec:a", "libmp3lame", "-f", "mp3", str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    processes = []
    original_popen = subprocess.Popen

    def track_process(command, *args, **kwargs):
        processes.append(command)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(audio_segment.subprocess, "Popen", track_process)
    monkeypatch.setattr(utils, "Popen", track_process)

    audio = audio_io.load_audio_file(path)

    assert abs(len(audio) - 1000) <= 100
    assert audio.frame_rate == 16_000
    assert any("ffprobe" in str(command[0]) for command in processes)
    decoder = next(command for command in processes if "ffmpeg" in str(command[0]))
    assert "-f" not in decoder[: decoder.index("-i")]
    path.unlink()
