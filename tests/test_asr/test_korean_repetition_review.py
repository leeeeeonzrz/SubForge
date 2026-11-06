import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from subforge.core.asr import final_coverage as fc
from subforge.core.asr import korean_repetition_review as rr
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


def words(items, score=0.9):
    result = ASRData([
        ASRDataSeg(text, start, end, timestamp_granularity="word", language_code="ko")
        for text, start, end in items
    ])
    for segment in result.segments:
        segment.words[0].alignment_score = score
    return result


@pytest.mark.parametrize("text", [
    "어 " * 12, "어" * 20, "아니 " * 12,
    "그러니까" * 12, "오늘은 " + "아" * 20 + " 집에 가요",
    "중국어로 우우라고 " + "삐" * 50,
    "띠요 띠요띠요 " + "삐" * 50,
])
def test_severe_korean_loops_are_candidates_even_without_timestamp_holes(text):
    data = words([(text, 1000, 6000)])
    assert rr.repetition_ranges(data) == [rr.RepetitionRange(1, 6)]
    assert data.segments[0].text == text


@pytest.mark.parametrize("text", [
    "네", "응", "아", "어", "네 네 네", "아니 아니 아니", "진짜 진짜 좋아요",
    "아니요 싫어요 그렇게 하면 안 돼요", "반짝반짝 두근두근 소곤소곤",
    "삐용삐용 구급차 소리", "띠요 띠요띠요 삐삐삐",
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ", "ㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎ",
    "yes " * 20, "um " * 30, "はい " * 20, "そうそうそうそう", "ありがとうございます",
])
def test_short_replies_emphasis_reduplication_laughter_english_japanese_unchanged(text):
    assert not rr.repetition_ranges(words([(text, 0, 6000)]))


def test_separate_replies_do_not_accumulate_into_loop():
    assert not rr.repetition_ranges(words([
        ("네", index * 2000, index * 2000 + 300) for index in range(20)
    ]))


def test_cross_segment_loop_and_unicode_normalization():
    import unicodedata

    data = words([(unicodedata.normalize("NFD", "어"), i * 500, i * 500 + 400)
                  for i in range(20)])
    assert rr.repetition_ranges(data) == [rr.RepetitionRange(0, 9.9)]


def test_marking_does_not_delete_a_real_long_chant_and_is_idempotent():
    data = words([("사랑해 " * 20, 1000, 15000)])
    before = copy.deepcopy(data.segments)
    rr.mark_repetitions_for_review(data)
    rr.mark_repetitions_for_review(data)
    assert [s.__dict__ for s in data.segments] == [s.__dict__ for s in before]
    assert len(data.coverage_issues) == 1
    assert data.coverage_issues[0]["reason"] == "suspected_repetition"


def original():
    return words([
        *[(f"앞부분{i}", i * 500, i * 500 + 350) for i in range(20)],
        *[("어", 10000 + i * 500, 10400 + i * 500) for i in range(12)],
        *[(f"뒷부분{i}", 16000 + i * 500, 16350 + i * 500) for i in range(24)],
    ])


def repaired():
    data = original()
    data.segments = [s for s in data.segments if not 10000 <= s.start_time < 16000]
    data.segments.extend(words([
        (text, 10000 + i * 900, 10800 + i * 900)
        for i, text in enumerate(["오늘은", "여러분이", "보고", "싶어서", "방송을", "켰어요"])
    ]).segments)
    data.segments.sort(key=lambda s: s.start_time)
    return data


@pytest.fixture
def setup(monkeypatch):
    from subforge.core.asr import silero_vad, ten_vad

    for backend in (silero_vad, ten_vad):
        monkeypatch.setattr(backend, "is_available", lambda: True)
        monkeypatch.setattr(backend, "run_vad_inference",
                            lambda samples, **kwargs: [(0, len(samples) // 16)])
    samples = np.zeros(40 * 16000, dtype=np.float32)
    context = SimpleNamespace(audio_path="original.wav", samples=lambda: samples)
    config = SimpleNamespace(cancel_event=None, whisperx_model="existing-model")
    records = []
    monkeypatch.setattr(fc, "_audit", lambda *a: [])
    monkeypatch.setattr(fc, "_save_evidence", lambda r: records.extend(r))
    return context, config, records


def run(data, setup):
    context, config, _ = setup
    return fc.repair_final_coverage(data, "original.wav", config, lambda *a: None, None, context)


def test_same_language_loop_recovers_with_two_different_windows_and_preserves_context(monkeypatch, setup):
    calls = []

    def decode(*args):
        calls.append((args[1], args[2]))
        return repaired()

    monkeypatch.setattr(fc, "_decode", decode)
    before = original()
    result = run(original(), setup)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert fc._words(result) == fc._words(repaired())
    assert not result.coverage_issues
    assert result.segments[0].__dict__ == before.segments[0].__dict__
    assert result.segments[-1].__dict__ == before.segments[-1].__dict__
    assert setup[2][0]["status"] == "recovered"
    assert setup[2][0]["gap"]["reason"] == "suspected_repetition"


@pytest.mark.parametrize("failure", ["disagreement", "repeat", "weak_alignment", "missing_anchors", "silence", "no_silero", "lost_reply"])
def test_unsafe_repetition_repair_keeps_every_original_word(monkeypatch, setup, failure):
    from subforge.core.asr import silero_vad

    calls = []
    data = original()
    if failure == "lost_reply":
        # A genuine short response shares the same suspect subtitle region.
        data.segments[20].text = "네 어"
    if failure == "silence":
        monkeypatch.setattr(silero_vad, "run_vad_inference", lambda *a, **k: [])
    if failure == "no_silero":
        monkeypatch.setattr(silero_vad, "is_available", lambda: False)

    def decode(*args):
        calls.append(1)
        candidate = original() if failure == "repeat" else repaired()
        if failure == "disagreement" and len(calls) == 2:
            candidate.segments[22].text = "달라졌어요"
        if failure == "weak_alignment":
            candidate.segments[22].words[0].alignment_score = 0.2
        if failure == "missing_anchors":
            candidate.segments = [s for s in candidate.segments if 10000 <= s.start_time < 16000]
        return candidate

    monkeypatch.setattr(fc, "_decode", decode)
    before = fc._words(data)
    result = run(data, setup)
    assert fc._words(result) == before
    assert result.coverage_issues[0]["reason"] == "suspected_repetition"
    assert len(calls) <= 2


def test_windows_without_ten_can_use_silero_plus_separate_alignment(monkeypatch, setup):
    from subforge.core.asr import ten_vad

    monkeypatch.setattr(ten_vad, "is_available", lambda: False)
    monkeypatch.setattr(ten_vad, "run_vad_inference", lambda *a, **k: pytest.fail("TEN unavailable"))
    monkeypatch.setattr(fc, "_decode", lambda *a: repaired())
    result = run(original(), setup)
    assert fc._words(result) == fc._words(repaired())
    assert not result.coverage_issues
    assert setup[2][0]["repetition_evidence"]["vad_backends"] == ["silero"]


def test_shortening_filler_only_is_not_a_valid_replacement(setup):
    context = setup[0]
    with pytest.raises(ValueError, match="substantial"):
        rr.validate_repetition_replacement(
            words([("어" * 20, 1000, 5000)]).segments,
            words([("어", 1000, 1100), ("아", 1500, 1600), ("음", 1800, 1900)]).segments,
            rr.RepetitionRange(1, 5), context,
        )


@pytest.mark.parametrize("expression", ["아니", "네", "응", "안", "못"])
def test_repeated_negation_or_confirmation_cannot_disappear_from_recovery(setup, expression):
    with pytest.raises(ValueError, match="repeated expression"):
        rr.validate_repetition_replacement(
            words([(f"{expression} " * 12, 10000, 15900)]).segments,
            [s for s in repaired().segments if 10000 <= s.start_time < 16000],
            rr.RepetitionRange(10, 15.9), setup[0],
        )


@pytest.mark.parametrize("false_support", ["inside_word", "neighboring_reply"])
def test_semantic_reply_requires_a_whole_word_within_disputed_audio(setup, false_support):
    source = words([("네 " * 12, 10000, 15900)]).segments
    replacement = [s for s in repaired().segments if 10000 <= s.start_time < 16000]
    if false_support == "inside_word":
        replacement[0].text = "동네에서"
    else:
        prefix = words([("네", 8000, 8200), ("오늘", 8500, 8900)]).segments
        source = prefix + source
        replacement = copy.deepcopy(prefix) + replacement
    with pytest.raises(ValueError, match="repeated expression"):
        rr.validate_repetition_replacement(
            source, replacement, rr.RepetitionRange(10, 15.9), setup[0],
        )


def test_real_speech_gap_keeps_budget_priority_over_earlier_repetition(monkeypatch, setup):
    from subforge.core.asr.whisperx_asr import _SpeechBackedGap

    gap = _SpeechBackedGap(20, 23, 2.5, 0.83, True)
    monkeypatch.setattr(fc, "_audit", lambda *args: [gap])
    monkeypatch.setattr(fc, "MAX_WINDOWS", 1)
    calls = []

    def decode(_context, left, right, *args):
        calls.append((left, right))
        raise ValueError("Evidence unavailable in this test")

    monkeypatch.setattr(fc, "_decode", decode)
    data = words([("어 " * 20, 2000, 4000), ("마지막", 30000, 30500)])
    run(data, setup)
    assert calls == [(10000, 33000)]
    assert setup[2][0]["gap"]["start"] == 20
    assert setup[2][1]["gap"]["reason"] == "suspected_repetition"
    assert setup[2][1]["status"] == "budget_exhausted"


def test_loop_budget_is_shared_with_existing_speech_recovery(monkeypatch, setup):
    monkeypatch.setattr(fc, "MAX_DECODE_SECONDS", 1)
    monkeypatch.setattr(fc, "_decode", lambda *a: pytest.fail("budget ignored"))
    before = fc._words(original())
    result = run(original(), setup)
    assert fc._words(result) == before
    assert result.coverage_issues[0]["reason"] == "suspected_repetition"
    assert setup[2][0]["status"] == "budget_exhausted"


def test_loop_cancellation_is_not_swallowed(monkeypatch, setup):
    event = threading.Event()
    setup[1].cancel_event = event

    def decode(*args):
        event.set()
        raise RuntimeError("stopped")

    monkeypatch.setattr(fc, "_decode", decode)
    with pytest.raises(RuntimeError, match="cancelled"):
        run(original(), setup)


def test_sentence_timestamps_flag_instead_of_inventing_word_alignment(monkeypatch, setup):
    data = ASRData([ASRDataSeg("어" * 20, 1000, 6000, timestamp_granularity="sentence")])
    monkeypatch.setattr(fc, "_decode", lambda *a: pytest.fail("sentence decode"))
    assert run(data, setup).coverage_issues[0]["reason"] == "suspected_repetition"


def test_repetition_message_describes_audio_review():
    from subforge.core.asr.speech_gap_repair import coverage_issue_message

    assert "Subtitle audio evidence" in coverage_issue_message([
        {"start": 1, "end": 5, "reason": "suspected_repetition"}
    ])
