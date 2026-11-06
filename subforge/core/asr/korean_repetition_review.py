"""Nominate severe Korean decoder loops without deleting repeated speech.

Text patterns are review triggers only. Automatic recovery additionally needs
the original audio, agreeing local decodes and the surrounding lexical anchors.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from .asr_data import ASRData

# A dozen cycles excludes ordinary replies, emphasis and Korean reduplication.
# Match syllables, not laughter jamo (ㅋㅋ/ㅎㅎ), and tolerate ASR spacing.
_LOOP = re.compile(r"([가-힣]{1,4}?)\1{11,}")
_SEMANTIC_SHORT_EXPRESSIONS = {"네", "예", "응", "안", "못"}


def _key(text: str) -> str:
    return "".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


@dataclass(frozen=True)
class RepetitionRange:
    start: float
    end: float
    reason: str = "suspected_repetition"


def repetition_ranges(data: ASRData) -> list[RepetitionRange]:
    """Find local syllable loops even when timestamps already cover the audio."""
    groups: list[list] = []
    for segment in data.segments:
        if not groups or segment.start_time - groups[-1][-1].end_time > 1200:
            groups.append([])
        groups[-1].append(segment)
    result: list[RepetitionRange] = []
    for group in groups:
        text, owners = [], []
        for index, segment in enumerate(group):
            key = _key(segment.text)
            text.append(key)
            owners.extend([index] * len(key))
        for match in _LOOP.finditer("".join(text)):
            first, last = group[owners[match.start()]], group[owners[match.end() - 1]]
            start, end = first.start_time / 1000, last.end_time / 1000
            if end <= start:
                continue
            if result and start <= result[-1].end:
                result[-1] = RepetitionRange(result[-1].start, max(end, result[-1].end))
            else:
                result.append(RepetitionRange(start, end))
    return result


def mark_repetitions_for_review(data: ASRData) -> ASRData:
    """Keep original text for engines without safe local acoustic recovery."""
    for region in repetition_ranges(data):
        issue = {"start": region.start, "end": region.end, "reason": region.reason}
        if issue not in data.coverage_issues:
            data.coverage_issues.append(issue)
    return data


def _overlap(start: int, end: int, ranges: list[tuple[int, int]]) -> int:
    total, last = 0, start
    for left, right in sorted(ranges):
        left, right = max(start, left), min(end, right)
        if right > left:
            total += max(0, right - max(last, left))
            last = max(last, right)
    return total


def validate_repetition_replacement(original, replacement, region, context) -> dict:
    """Reject speculative cleanup, lost context, silence and weak alignment.

    Two identical ASR answers alone cannot prove a loop is false. Require diverse
    new Korean speech aligned by the separate forced-alignment model and
    supported by Silero VAD. Check TEN-VAD too when the platform provides it.
    Never replace a long loop with an empty span or merely fewer filler tokens.
    """
    old_text = _key(" ".join(segment.text for segment in original))
    new_text = _key(" ".join(segment.text for segment in replacement))
    if not _LOOP.search(old_text) or _LOOP.search(new_text):
        raise ValueError("Local recognition still contains repeated speech")
    cursor = 0
    # Everything outside the suspect cycles must survive, including short
    # answers, names, numbers and emphasis in the same subtitle cue.
    for piece in _LOOP.split(old_text)[::2]:
        if piece:
            cursor = new_text.find(piece, cursor)
            if cursor < 0:
                raise ValueError("Repetition recovery would lose original content")
            cursor += len(piece)
    disputed = [
        segment for segment in replacement
        if segment.start_time < region.end * 1000 and segment.end_time > region.start * 1000
    ]
    keys = [_key(segment.text) for segment in disputed]
    # Keep meaningful repeated expressions within the disputed audio itself.
    # A neighboring reply, or the 네 inside 동네, cannot preserve a lost "yes".
    for match in _LOOP.finditer(old_text):
        expression = match[1]
        if expression in _SEMANTIC_SHORT_EXPRESSIONS:
            preserved = expression in keys
        elif len(expression) > 1:
            preserved = any(expression in key for key in keys)
        else:
            continue
        if not preserved:
            raise ValueError("Repetition recovery would lose a repeated expression")
    lexical = [key for key in keys if re.search(r"[가-힣]", key)]
    if len(set(lexical)) < 3 or sum(len(key) for key in lexical) < 6:
        raise ValueError("Repeated speech has no substantial corroborated replacement")
    scores = []
    for segment in disputed:
        if not segment.words:
            raise ValueError("Repetition recovery lacks word alignment evidence")
        for word in segment.words:
            score = word.alignment_score
            if score is None or not math.isfinite(score) or score < 0.45:
                raise ValueError("Repetition recovery has weak word alignment")
            scores.append(score)
    if not scores or sum(scores) / len(scores) < 0.65:
        raise ValueError("Repetition recovery has weak aggregate alignment")

    from . import silero_vad, ten_vad

    if not silero_vad.is_available():
        raise ValueError("Speech detector unavailable for repetition recovery")
    backends = [(silero_vad, 0.35, "silero")]
    if ten_vad.is_available():
        backends.append((ten_vad, 0.5, "ten"))
    left = max(0, round(region.start * 1000) - 500)
    right = min(len(context.samples()) // 16, round(region.end * 1000) + 500)
    samples = context.samples()[left * 16:right * 16]
    for backend, threshold, _name in backends:
        ranges = backend.run_vad_inference(
            samples, sample_rate=16000, threshold=threshold,
            min_speech_ms=100, min_silence_ms=180, speech_pad_ms=0,
        )
        ranges = [(a + left, b + left) for a, b in ranges]
        for segment in disputed:
            start, end = max(left, segment.start_time), min(right, segment.end_time)
            if end <= start or _overlap(start, end, ranges) < (end - start) * 0.3:
                raise ValueError("Repetition replacement lacks speech support")
    return {
        "vad_backends": [name for _backend, _threshold, name in backends],
        "minimum_alignment_score": min(scores),
        "mean_alignment_score": sum(scores) / len(scores),
    }
