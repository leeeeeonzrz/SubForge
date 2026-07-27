"""Backend subtitle conversion helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.api.subtitles import (
    SaveRequest,
    _normalize_segment_timing,
    _timestamp_to_ms,
    save_subtitle,
    segments_to_ass,
    segments_to_srt,
    segments_to_txt,
    segments_to_vtt,
)

from subforge.core.utils.atomic_write import encode_srt_text


def test_save_writes_original_translated_and_bilingual_files(tmp_path, monkeypatch):
    import asyncio

    import app.api.subtitles as subtitles_module

    source = tmp_path / "episode_bilingual.srt"
    source.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(subtitles_module, "validate_path", lambda path: Path(path))
    segments = [
        {
            "id": 1,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "Hello world.",
            "translated": "你好，世界。",
        }
    ]

    result = asyncio.run(
        save_subtitle(SaveRequest(file_path=str(source), segments=segments))
    )

    original = tmp_path / "episode_original.srt"
    translated = tmp_path / "episode_translated.srt"
    bilingual = tmp_path / "episode_bilingual.srt"
    assert set(result["files"]) == {"original", "translated", "bilingual"}
    assert result["file_path"] == str(bilingual)
    assert result["directory"] == str(tmp_path)
    assert "Hello world." in original.read_text(encoding="utf-8")
    assert "你好，世界。" not in original.read_text(encoding="utf-8")
    assert "你好，世界。" in translated.read_text(encoding="utf-8")
    assert "Hello world." not in translated.read_text(encoding="utf-8")
    bilingual_text = bilingual.read_text(encoding="utf-8")
    assert "你好，世界。\nHello world." in bilingual_text
    assert not (tmp_path / "episode_bilingual_original.srt").exists()


def test_normalize_segment_timing_removes_overlap():
    segments = [
        {"id": 1, "start": "00:00:00.000", "end": "00:00:02.000", "text": "First"},
        {"id": 2, "start": "00:00:01.500", "end": "00:00:03.000", "text": "Second"},
    ]

    normalized = _normalize_segment_timing(segments)

    assert normalized[0]["end"] == "00:00:01.750"
    assert normalized[1]["start"] == "00:00:01.750"


def test_segments_to_srt_uses_normalized_timing_shape():
    segments = _normalize_segment_timing([
        {"id": 1, "start": "00:00:00,000", "end": "00:00:02,000", "text": "First"},
        {"id": 2, "start": "00:00:01,500", "end": "00:00:03,000", "text": "Second"},
    ])

    srt = segments_to_srt(segments)

    assert "00:00:00,000 --> 00:00:01,750" in srt
    assert "00:00:01,750 --> 00:00:03,000" in srt


def test_bilingual_exports_put_translation_above_source():
    segments = [
        {
            "id": 1,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "This chart shows jet fuel costs.",
            "translated": "这张图展示的是航空燃油成本。",
        }
    ]

    srt = segments_to_srt(segments)
    vtt = segments_to_vtt(segments)
    txt = segments_to_txt(segments)
    ass = segments_to_ass(segments)

    assert "这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs." in srt
    assert "这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs." in vtt
    assert txt.startswith("这张图展示的是航空燃油成本。\nThis chart shows jet fuel costs.")
    assert "这张图展示的是航空燃油成本。\\NThis chart shows jet fuel costs." in ass


def test_srt_download_encoding_is_word_compatible():
    payload = encode_srt_text("1\n00:00:00,000 --> 00:00:01,000\n中文\n")

    assert payload.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in payload
    assert payload.decode("utf-8-sig").splitlines()[-1] == "中文"


@pytest.mark.parametrize("value", ["", "not-a-time", "00:61:00.000", -1, float("nan")])
def test_strict_timestamp_parser_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _timestamp_to_ms(value, strict=True)


def test_strict_timestamp_parser_keeps_supported_formats():
    assert _timestamp_to_ms("00:01:02,345", strict=True) == 62_345
    assert _timestamp_to_ms("1:02.34", strict=True) == 62_340
