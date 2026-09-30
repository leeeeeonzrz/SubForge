import wave
from types import SimpleNamespace

import pytest

from subforge.core.asr import base as asr_base
from subforge.core.asr.asr_data import ASRDataSeg
from subforge.core.asr.base import BaseASR


class FakeCache:
    def __init__(self):
        self.get_calls = 0
        self.set_calls = 0
        self.values = {}

    def get(self, key, default=None):
        self.get_calls += 1
        return self.values.get(key, default)

    def set(self, key, value, expire=None):
        self.set_calls += 1
        self.values[key] = value


class DummyASR(BaseASR):
    def _run(self, callback=None, **kwargs):
        return "fresh"

    def _make_segments(self, resp_data):
        return [ASRDataSeg(str(resp_data), 0, 1000)]


def test_duration_keeps_wav_path_to_avoid_per_segment_ffmpeg(monkeypatch, tmp_path):
    audio_path = tmp_path / "vad_0001.wav"
    audio_path.write_bytes(b"RIFF-test")
    received = []
    monkeypatch.setattr(
        asr_base.AudioSegment,
        "from_file",
        lambda source: received.append(source)
        or SimpleNamespace(duration_seconds=1.25),
    )
    asr = BaseASR.__new__(BaseASR)
    asr.audio_input = str(audio_path)
    asr.file_binary = audio_path.read_bytes()

    assert asr._get_audio_duration() == 1.25
    assert received == [str(audio_path)]


def test_wav_duration_does_not_spawn_ffmpeg(monkeypatch, tmp_path):
    from pydub import audio_segment

    audio_path = tmp_path / "vad_0002.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 16_000)

    monkeypatch.setattr(asr_base, "get_asr_cache", FakeCache)

    def unexpected_process(*_args, **_kwargs):
        raise AssertionError("WAV duration lookup must not start ffmpeg")

    monkeypatch.setattr(audio_segment.subprocess, "Popen", unexpected_process)

    asr = DummyASR(str(audio_path), use_cache=False)

    assert asr.audio_duration == 1.0


def test_base_asr_does_not_read_or_write_cache_when_disabled(monkeypatch, tmp_path):
    from pydub.generators import Sine

    audio_path = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=500).export(audio_path, format="wav").close()
    fake_cache = FakeCache()
    fake_cache.values["asr-v2:DummyASR:any"] = "cached"

    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: fake_cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)

    asr = DummyASR(str(audio_path), use_cache=False)
    asr._get_key = lambda: "any"
    result = asr.run()

    assert result.segments[0].text == "fresh"
    assert fake_cache.get_calls == 0
    assert fake_cache.set_calls == 0


def test_base_asr_uses_cache_only_when_enabled(monkeypatch, tmp_path):
    from pydub.generators import Sine

    audio_path = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=500).export(audio_path, format="wav").close()
    fake_cache = FakeCache()
    fake_cache.values["asr-v2:DummyASR:any"] = "cached"

    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: fake_cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)

    asr = DummyASR(str(audio_path), use_cache=True)
    asr._get_key = lambda: "any"
    result = asr.run()

    assert result.segments[0].text == "cached"
    assert fake_cache.get_calls == 1
    assert fake_cache.set_calls == 0


@pytest.mark.parametrize("cached", [False, True])
def test_base_asr_preserves_coverage_issues(monkeypatch, tmp_path, cached):
    from pydub.generators import Sine

    audio = tmp_path / "audio.wav"
    Sine(440).to_audio_segment(duration=50).export(audio, format="wav").close()
    payload = {"coverage_issues": [{"start": 10, "end": 14.2, "reason": "decode_budget"}]}
    cache = FakeCache()
    cache.values["asr-v2:DummyASR:any"] = payload
    monkeypatch.setattr(asr_base, "get_asr_cache", lambda: cache)
    monkeypatch.setattr(asr_base, "is_cache_enabled", lambda: True)
    asr = DummyASR(str(audio), use_cache=cached)
    asr._get_key = lambda: "any"
    asr._run = lambda *args, **kwargs: payload
    assert asr.run().coverage_issues == payload["coverage_issues"]
