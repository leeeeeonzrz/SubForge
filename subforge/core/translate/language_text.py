"""Small Unicode helpers shared by translation checks and offline evaluation."""

from __future__ import annotations

import re
import unicodedata

_HANGUL = r"\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff"
HANGUL_RE = re.compile(f"[{_HANGUL}]")
HANGUL_WORD_RE = re.compile(f"[{_HANGUL}]+")
_KOREAN_SPAN_RE = re.compile(rf"[{_HANGUL}]+(?:[ \t]+[{_HANGUL}]+)*")
_SOURCE_TOKEN_RE = re.compile(rf"[A-Za-z0-9']+|[{_HANGUL}]+")


def normalized_text(text: str) -> str:
    """Treat composed and decomposed Hangul as the same text, without rewriting it."""
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def compact_similarity_text(text: str) -> str:
    """Keep letters and numbers from every script when comparing subtitles."""
    return "".join(char for char in normalized_text(text) if char.isalnum())


def source_tokens(text: str) -> list[str]:
    """Add Korean word blocks to the existing English word-based heuristics.

    Japanese clauses cannot be treated as single words by those heuristics:
    doing so would newly reject natural paraphrased repetition. Script-neutral
    comparison uses compact_similarity_text instead.
    """
    return _SOURCE_TOKEN_RE.findall(normalized_text(text))


def has_korean_clause_residue(output: str, source: str) -> bool:
    """Find substantial copied Korean clauses, while allowing isolated Korean names.

    This is a retry signal, never a deletion rule. A shared name or short quoted
    word alone cannot prove a translation is missing. Require copied wording and
    either a sentence ending or a long multiword span.
    """
    source_words = set(HANGUL_WORD_RE.findall(normalized_text(source)))
    # Chinese title brackets explicitly identify a named work; a Korean song
    # title can itself look like a sentence. Leave title translation to context.
    prose_output = re.sub(r"《[^》]*》", "", normalized_text(output))
    for span in _KOREAN_SPAN_RE.findall(prose_output):
        # Chinese connectors separate retained names from one another. Counting
        # all Hangul in a translated sentence would mistake a list of names for
        # an untranslated clause (for example 이루다和김민지).
        output_words = HANGUL_WORD_RE.findall(span)
        copied = [word for word in output_words if word in source_words]
        if not copied:
            continue
        copied_chars = sum(len(word) for word in copied)
        last_word = output_words[-1]
        sentence_ending = (
            last_word in source_words
            and len(last_word) >= 3
            and re.search(
                r"(?:요|니다|했어|했지|했네|할게|할까|잖아|거든|는데|[했었았겠]다|한다|된다|하다|는다)$",
                last_word,
            )
        )
        if (
            (len(copied) >= 2 and copied_chars >= 6 and sentence_ending)
            or (len(copied) == 1 and copied_chars >= 8 and sentence_ending)
            or (
                len(copied) >= 3
                and copied_chars >= 14
                and any(word in {"저는", "제가", "나는", "내가", "우리는", "저희는"} for word in copied)
            )
        ):
            return True
    return False
