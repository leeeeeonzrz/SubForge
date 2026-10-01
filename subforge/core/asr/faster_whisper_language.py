"""CTranslate2 adapter for conservative WhisperX language-switch recovery.

The already loaded WhisperX model is reused; no second GPU model or MLX runtime
is needed. Only fully validated local replacements are committed.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable

SAMPLE_RATE = 16_000
CONTEXT_SECONDS = 0.75


class FasterWhisperLanguageDecoder:
    def __init__(self, model: Any, audio: Any, check_cancelled: Callable[[str], None]):
        self.model = model
        self.audio = audio
        self.check_cancelled = check_cancelled
        self.is_multilingual = bool(model.model.is_multilingual)

    def probabilities(self, start: float, end: float) -> dict[str, float]:
        self.check_cancelled("language probing")
        clip = self.audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        _, _, probabilities = self.model.detect_language(audio=clip, vad_filter=False)
        return {str(language): float(probability) for language, probability in probabilities}

    def decode(self, clip: Any, language: str | None = None) -> dict:
        self.check_cancelled("language re-transcription")
        segments, info = self.model.transcribe(
            clip,
            language=language,
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        decoded = []
        for segment in segments:
            self.check_cancelled("language re-transcription")
            decoded.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                    "words": [
                        {"start": word.start, "end": word.end, "word": word.word}
                        for word in segment.words or []
                    ],
                }
            )
        return {"segments": decoded, "language": info.language}

    def _localized(
        self,
        start: float,
        end: float,
        language: str,
        *,
        source_text: str = "",
    ) -> list[dict]:
        from .whisperx_asr import _usable_mlx_recovery_segment

        context_start = max(0.0, start - CONTEXT_SECONDS)
        context_end = min(len(self.audio) / SAMPLE_RATE, end + CONTEXT_SECONDS)
        decoded = self.decode(
            self.audio[int(context_start * SAMPLE_RATE) : int(context_end * SAMPLE_RATE)],
            language,
        )
        localized: list[dict] = []
        for segment in decoded.get("segments") or []:
            if not _usable_mlx_recovery_segment(segment, speech_ratio=0.0):
                continue
            # Trim by real word timestamps; fully external context words are
            # never copied or borrowed from the neighbouring Korean cue.
            words = []
            for word in segment.get("words") or []:
                left = float(word["start"]) + context_start
                right = float(word["end"]) + context_start
                if not (math.isfinite(left) and math.isfinite(right) and right > left):
                    continue
                owned = start <= (left + right) / 2 < end
                # A word onset can be timestamped into the preceding speech.
                # Keep a straddling word only when the existing
                # source cue confirms that exact word belongs here.
                token = str(word["word"]).strip()
                source_backed_edge = (
                    bool(token and source_text and token in source_text)
                    and left < end
                    and right > start
                )
                if owned or source_backed_edge:
                    left, right = max(start, left), min(end, right)
                    if words and left < words[-1]["end"]:
                        return []
                    words.append({**word, "start": left, "end": right})
            if not words:
                continue
            text = "".join(word["word"] for word in words).strip()
            if text:
                localized.append(
                    {
                        **segment,
                        "start": words[0]["start"],
                        "end": words[-1]["end"],
                        "text": text,
                        "words": words,
                        "language": language,
                    }
                )
        localized.sort(key=lambda item: item["start"])
        if any(left["end"] > right["start"] for left, right in zip(localized, localized[1:])):
            return []
        return localized

    def _extend_into_uncovered_edges(self, candidate: Any, segments: list[dict]) -> Any:
        """Recover window-edge words only in gaps with independent language evidence."""
        start, end = candidate.start, candidate.end
        if not any(item["start"] <= start < item["end"] for item in segments):
            preceding = max((item["end"] for item in segments if item["end"] <= start), default=0.0)
            earlier = max(preceding, start - 2.0)
            if start - earlier >= 0.75:
                probabilities = self.probabilities(earlier, start)
                if probabilities.get(candidate.language, 0.0) >= 0.80:
                    start = earlier
        if not any(item["start"] < end <= item["end"] for item in segments):
            following = min(
                (item["start"] for item in segments if item["start"] >= end),
                default=len(self.audio) / SAMPLE_RATE,
            )
            later = min(following, end + 2.0)
            if later - end >= 0.75:
                probabilities = self.probabilities(end, later)
                if probabilities.get(candidate.language, 0.0) >= 0.80:
                    end = later
        return replace(candidate, start=start, end=end)

    def replace_ranges(self, result: dict, ranges: list[Any], primary_language: str) -> dict:
        from .whisperx_asr import _subtract_language_ranges

        combined = [
            {**segment, "language": primary_language}
            for segment in result.get("segments") or []
            if isinstance(segment, dict)
        ]
        for language_range in ranges:
            self.check_cancelled("language replacement")
            language_range = self._extend_into_uncovered_edges(language_range, combined)
            overlapping = [
                segment
                for segment in combined
                if segment["end"] > language_range.start and segment["start"] < language_range.end
            ]
            foreign = self._localized(
                language_range.start,
                language_range.end,
                language_range.language,
                source_text=" ".join(str(segment.get("text") or "") for segment in overlapping),
            )
            if not foreign:
                continue
            primary_parts: list[dict] = []
            safe = True
            for segment in overlapping:
                # Never cut a replacement already committed for a different range.
                if segment["language"] != primary_language:
                    safe = False
                    break
                for start, end in _subtract_language_ranges(
                    segment["start"],
                    segment["end"],
                    [language_range],
                ):
                    # A short or undecodable remainder is not evidence of silence.
                    # Roll back this entire switch instead of dropping primary speech.
                    if end - start < 0.75:
                        safe = False
                        break
                    parts = self._localized(
                        start,
                        end,
                        primary_language,
                        source_text=str(segment.get("text") or ""),
                    )
                    if not parts:
                        safe = False
                        break
                    primary_parts.extend(parts)
                if not safe:
                    break
            if safe:
                replaced_ids = {id(segment) for segment in overlapping}
                combined = [segment for segment in combined if id(segment) not in replaced_ids]
                combined.extend(primary_parts + foreign)
        combined.sort(key=lambda item: (item["start"], item["end"]))
        return {
            **result,
            "segments": combined,
            "languages": sorted({item["language"] for item in combined}),
        }
