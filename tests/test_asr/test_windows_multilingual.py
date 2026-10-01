import json
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from subforge.core.asr.faster_whisper_language import FasterWhisperLanguageDecoder
from subforge.core.asr.whisperx_asr import WhisperXASR, _LanguageRange


def make_asr():
    asr = WhisperXASR.__new__(WhisperXASR)
    asr.language = "ko"
    asr.detect_additional_languages = True
    asr.cancel_event = threading.Event()
    asr.align_model = "example/korean-alignment-model"
    asr.model_dir = "unused"
    asr.missing_alignment_model_callback = None
    return asr


def make_decoder():
    model = SimpleNamespace(model=SimpleNamespace(is_multilingual=True))
    return FasterWhisperLanguageDecoder(model, np.zeros(30 * 16000), lambda _: None)


def test_adapter_reuses_model_and_requests_original_language_words():
    calls = []
    segment = SimpleNamespace(
        start=1.0,
        end=2.0,
        text=" 한국어",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        compression_ratio=1.1,
        words=[SimpleNamespace(start=1.0, end=2.0, word=" 한국어")],
    )
    model = SimpleNamespace(
        model=SimpleNamespace(is_multilingual=True),
        detect_language=lambda **kw: ("ko", 0.99, [("ko", 0.99), ("en", 0.01)]),
        transcribe=lambda clip, **kw: (
            calls.append(kw) or iter([segment]),
            SimpleNamespace(language="ko"),
        ),
    )
    decoder = FasterWhisperLanguageDecoder(model, np.zeros(48000), lambda _: None)
    assert decoder.probabilities(0, 2) == {"ko": 0.99, "en": 0.01}
    result = decoder.decode(np.zeros(32000), "ko")
    assert result["segments"][0]["words"][0]["word"] == " 한국어"
    assert calls[0]["language"] == "ko"
    assert calls[0]["task"] == "transcribe"
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["condition_on_previous_text"] is False


def test_shared_detector_accepts_sustained_korean_and_rejects_weak_evidence(monkeypatch):
    asr = make_asr()
    audio = np.zeros(24 * 16000)
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr._detect_speech_in_mlx_gaps", lambda *a: [(0, 24000)]
    )
    result = {"segments": [{"start": 0.0, "end": 24.0, "text": "primary transcript"}]}

    def decode(clip):
        return {
            "language": "ko",
            "segments": [
                {
                    "start": 0,
                    "end": 10,
                    "text": "한국어 발화입니다 정말 감사합니다",
                    "avg_logprob": -0.2,
                }
            ],
        }

    ranges = asr._detect_language_ranges(
        audio, result, "en", lambda *a: {"ko": 0.99, "en": 0.01}, decode
    )
    assert ranges and {item.language for item in ranges} == {"ko"}
    assert (
        asr._detect_language_ranges(
            audio, result, "en", lambda *a: {"ko": 0.55, "en": 0.45}, decode
        )
        == []
    )


def test_detector_skips_silence_without_decoding(monkeypatch):
    monkeypatch.setattr("subforge.core.asr.whisperx_asr._detect_speech_in_mlx_gaps", lambda *a: [])

    def forbidden(*args):
        pytest.fail("Silent audio must not be decoded as a language switch")

    assert (
        make_asr()._detect_language_ranges(
            np.zeros(16000),
            {"segments": [{"start": 0, "end": 1, "text": "text"}]},
            "ko",
            forbidden,
            forbidden,
        )
        == []
    )


def test_language_probe_cancel_is_not_swallowed(monkeypatch):
    asr = make_asr()
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr._detect_speech_in_mlx_gaps", lambda *a: [(0, 24000)]
    )

    def decode(clip):
        asr.cancel_event.set()
        asr._raise_if_cancelled("test")

    with pytest.raises(RuntimeError, match="cancelled"):
        asr._detect_language_ranges(
            np.zeros(24 * 16000),
            {"segments": [{"start": 0, "end": 24, "text": "original"}]},
            "en",
            lambda *a: {"ko": 0.99, "en": 0.01},
            decode,
        )


def test_localization_keeps_korean_spaces_and_excludes_context_words(monkeypatch):
    decoder = make_decoder()
    # Context starts at 9.25; only the words inside 10..12 belong to the new cue.
    monkeypatch.setattr(
        decoder,
        "decode",
        lambda *a: {
            "segments": [
                {
                    "start": 0,
                    "end": 4,
                    "text": " context 한국어 문장입니다 outside",
                    "avg_logprob": -0.1,
                    "words": [
                        {"start": 0, "end": 0.5, "word": " context"},
                        {"start": 1, "end": 1.5, "word": " 한국어"},
                        {"start": 1.6, "end": 2.1, "word": " 문장입니다"},
                        {"start": 3, "end": 3.5, "word": " outside"},
                    ],
                }
            ]
        },
    )
    localized = decoder._localized(10, 12, "ko")
    assert localized[0]["text"] == "한국어 문장입니다"
    assert localized[0]["start"] == 10.25
    assert localized[0]["end"] == 11.35
    assert localized[0]["language"] == "ko"


@pytest.mark.parametrize("failure", ["empty_foreign", "empty_primary", "short_primary"])
def test_replacement_rolls_back_without_erasing_original(monkeypatch, failure):
    decoder = make_decoder()
    start = 9.5 if failure == "short_primary" else 5.0
    original = {"start": start, "end": 20.0, "text": "보존할 한국어 원문"}

    def local(start, end, language, **_kwargs):
        if (failure == "empty_foreign" and language == "en") or (
            failure == "empty_primary" and language == "ko"
        ):
            return []
        return [{"start": start, "end": end, "text": language, "language": language}]

    monkeypatch.setattr(decoder, "_localized", local)
    output = decoder.replace_ranges(
        {"segments": [original]}, [_LanguageRange(10, 15, "en", 0.99)], "ko"
    )
    assert output["segments"] == [{**original, "language": "ko"}]
    assert original == {"start": start, "end": 20.0, "text": "보존할 한국어 원문"}


def test_replacement_preserves_both_primary_sides_and_unaffected_cue(monkeypatch):
    decoder = make_decoder()
    monkeypatch.setattr(
        decoder,
        "_localized",
        lambda start, end, language, **_kwargs: [
            {"start": start, "end": end, "text": language, "language": language}
        ],
    )
    untouched = {"start": 0, "end": 4, "text": "안녕하세요"}
    output = decoder.replace_ranges(
        {"segments": [untouched, {"start": 5, "end": 20, "text": "mixed"}]},
        [_LanguageRange(10, 15, "en", 0.99)],
        "ko",
    )
    assert [(s["start"], s["end"], s["language"]) for s in output["segments"]] == [
        (0, 4, "ko"),
        (5, 10, "ko"),
        (10, 15, "en"),
        (15, 20, "ko"),
    ]
    assert output["segments"][0]["text"] == "안녕하세요"


@pytest.mark.parametrize(
    "decision, expected_ranges, expected_skip",
    [
        ("ignore", 0, set()),
        ("continue", 1, {"en"}),
        ("retry", 1, set()),
    ],
)
def test_missing_foreign_alignment_decisions(monkeypatch, decision, expected_ranges, expected_skip):
    asr = make_asr()
    ready = False

    def decide(models):
        nonlocal ready
        assert models[0]["language"] == "en"
        ready = decision == "retry"
        return decision

    asr.missing_alignment_model_callback = decide
    monkeypatch.setattr("subforge.core.asr.whisperx_asr.is_alignment_model_ready", lambda *a: ready)
    ranges, skip = asr._resolve_missing_alignment_models("ko", [_LanguageRange(10, 20, "en", 0.99)])
    assert len(ranges) == expected_ranges
    assert skip == expected_skip


def test_custom_korean_alignment_does_not_leak_into_english():
    asr = make_asr()
    assert asr._resolve_align_model_name("ko") == "example/korean-alignment-model"
    assert asr._resolve_align_model_name("en") == "WAV2VEC2_ASR_LARGE_LV60K_960H"


def test_foreign_only_group_uses_its_own_alignment(monkeypatch):
    asr = make_asr()
    called = []
    monkeypatch.setattr(
        asr,
        "_align_result",
        lambda result, audio, language, *args: called.append(language)
        or {"segments": result["segments"]},
    )
    output = asr._align_multilingual_result(
        {"segments": [{"start": 0, "end": 2, "text": "Hello", "language": "en"}]},
        [],
        "ko",
        lambda *a: None,
        None,
    )
    assert called == ["en"]
    assert output["segments"][0]["language"] == "en"


def test_cache_separates_model_and_language_supplement():
    asr = make_asr()
    asr.crc32_hex = "1234"
    asr.uses_mlx = False
    asr.whisper_model = "large-v3"
    asr.align_device = "cpu"
    asr.compute_type = "int8"
    asr.batch_size = 4
    asr.need_word_time_stamp = True
    first = asr._get_key()
    asr.detect_additional_languages = False
    assert asr._get_key() != first
    asr.detect_additional_languages = True
    asr.whisper_model = "tiny"
    assert asr._get_key() != first
    assert json.loads(first)["mixed_language_revision"] > 0


@pytest.mark.parametrize(
    "language, enabled, multilingual, expected",
    [
        ("ko", True, True, True),
        ("ko", False, True, False),
        (None, False, True, True),
        ("en", True, False, False),
    ],
)
def test_standard_whisperx_executes_supplement_only_when_requested(
    monkeypatch,
    tmp_path,
    language,
    enabled,
    multilingual,
    expected,
):
    asr = make_asr()
    asr.language = language
    asr.detect_additional_languages = enabled
    asr.audio_input = str(tmp_path / "audio.wav")
    asr.file_binary = None
    asr.whisper_model = "tiny"
    asr.transcribe_device = asr.align_device = "cpu"
    asr.compute_type = "int8"
    asr.batch_size = 1
    asr.segment_callback = None
    model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(is_multilingual=multilingual)),
        transcribe=lambda *a, **kw: {
            "language": language or "ko",
            "segments": [{"text": "원문", "start": 0, "end": 10}],
        },
    )
    for name in ("whisperx", "whisperx.alignment", "whisperx.asr", "whisperx.audio"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["whisperx.asr"].load_model = lambda *a, **kw: model
    sys.modules["whisperx.audio"].load_audio = lambda *a: np.zeros(160000)
    monkeypatch.setattr(
        "subforge.core.asr.whisperx_asr._install_offline_sentence_tokenizer", lambda *a: None
    )
    calls = []
    monkeypatch.setattr(
        asr,
        "_detect_language_ranges",
        lambda *a: calls.append("detect") or [_LanguageRange(2, 5, "en", 0.99)],
    )
    monkeypatch.setattr(
        asr, "_resolve_missing_alignment_models", lambda lang, ranges: (ranges, {"en"})
    )
    monkeypatch.setattr(
        FasterWhisperLanguageDecoder,
        "replace_ranges",
        lambda self, result, ranges, lang: {
            "segments": [{"text": "Hello", "start": 2, "end": 5, "language": "en"}]
        }
        if ranges
        else result,
    )
    monkeypatch.setattr(asr, "_align_multilingual_result", lambda result, *a: result)
    output = asr._run_standard()
    assert bool(calls) == expected
    assert output["segments"][0]["text"] == ("Hello" if expected else "원문")


@pytest.mark.parametrize(
    "field,value",
    [
        ("avg_logprob", -2.0),
        ("no_speech_prob", 0.95),
        ("compression_ratio", 4.0),
    ],
)
def test_low_quality_foreign_decode_keeps_original(monkeypatch, field, value):
    decoder = make_decoder()
    segment = {
        "start": 0,
        "end": 2,
        "text": " hallucination",
        "avg_logprob": -0.1,
        "words": [{"start": 1, "end": 2, "word": " hallucination"}],
        field: value,
    }
    monkeypatch.setattr(decoder, "decode", lambda *a: {"segments": [segment]})
    original = {"start": 10, "end": 15, "text": "원래 문장을 보존합니다"}
    output = decoder.replace_ranges(
        {"segments": [original]}, [_LanguageRange(10, 15, "en", 0.99)], "ko"
    )
    assert output["segments"] == [{**original, "language": "ko"}]


def test_local_decode_cancellation_propagates_before_model_inference():
    asr = make_asr()
    asr.cancel_event.set()
    decoder = FasterWhisperLanguageDecoder(
        SimpleNamespace(model=SimpleNamespace(is_multilingual=True)),
        np.zeros(16000),
        asr._raise_if_cancelled,
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        decoder.decode(np.zeros(16000), "en")


def test_primary_boundary_word_requires_source_support(monkeypatch):
    decoder = make_decoder()
    monkeypatch.setattr(
        decoder,
        "decode",
        lambda *a: {
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": " 이제 다시",
                    "avg_logprob": -0.1,
                    "words": [
                        {"start": 0, "end": 1, "word": " 이제"},
                        {"start": 1, "end": 2, "word": " 다시"},
                    ],
                }
            ]
        },
    )
    # At context_start=9.25 the first midpoint lies before start=10,
    # but the word overlaps the cue and is supported by the original Korean.
    assert decoder._localized(10, 12, "ko", source_text="이제 다시")[0]["text"] == "이제 다시"
    assert decoder._localized(10, 12, "ko")[0]["text"] == "다시"


@pytest.mark.parametrize("probability,expected_start", [(0.99, 8.0), (0.5, 10.0)])
def test_foreign_edges_expand_only_into_gaps_with_language_evidence(
    monkeypatch, probability, expected_start
):
    decoder = make_decoder()
    monkeypatch.setattr(decoder, "probabilities", lambda *a: {"en": probability})
    original = [{"start": 0, "end": 8}, {"start": 15, "end": 20}]
    candidate = decoder._extend_into_uncovered_edges(_LanguageRange(10, 16, "en", 0.99), original)
    assert candidate.start == expected_start
    assert candidate.end == 16  # Do not expand over existing main-language speech.


def test_existing_foreign_boundary_word_is_source_backed(monkeypatch):
    decoder = make_decoder()
    monkeypatch.setattr(
        decoder,
        "decode",
        lambda *a: {
            "segments": [
                {
                    "start": 0,
                    "end": 2,
                    "text": " Now I",
                    "avg_logprob": -0.1,
                    "words": [
                        {"start": 0, "end": 1, "word": " Now"},
                        {"start": 1, "end": 2, "word": " I"},
                    ],
                }
            ]
        },
    )
    original = {"start": 10, "end": 12, "text": "Now I"}
    output = decoder.replace_ranges(
        {"segments": [original]}, [_LanguageRange(10, 12, "en", 0.99)], "ko"
    )
    assert output["segments"][0]["text"] == "Now I"
    assert output["segments"][0]["language"] == "en"
