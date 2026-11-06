"""Korean conversational safety and Unicode coverage without external model calls."""

import unicodedata

import pytest

from subforge.core.entities import SubtitleProcessData
from subforge.core.translate.guidance import target_language_style_rules
from subforge.core.translate.language_text import compact_similarity_text, source_tokens
from subforge.core.translate.llm_translator import LLMTranslator
from subforge.core.translate.quality import (
    inspect_translation_batch,
    is_source_copy,
    is_untranslated_output,
)
from subforge.core.translate.types import TargetLanguage


@pytest.fixture
def translator():
    instance = LLMTranslator(
        thread_num=1, batch_num=1, target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        model="test", custom_prompt="", is_reflect=False, update_callback=None,
    )
    yield instance
    instance.stop()


def test_unicode_comparison_preserves_korean_japanese_and_english():
    assert compact_similarity_text("응, 오늘도 고마워요!") == "응오늘도고마워요"
    assert compact_similarity_text(unicodedata.normalize("NFD", "고마워요")) == "고마워요"
    assert compact_similarity_text("今日はありがとう！") == "今日はありがとう"
    assert compact_similarity_text("Hello, world!") == "helloworld"
    assert source_tokens("오늘 Weverse live 해요!") == ["오늘", "weverse", "live", "해요"]
    assert source_tokens("I don't know.") == ["i", "don't", "know"]


@pytest.mark.parametrize("source", ["오늘도 와 줘서 고마워요", "今日は来てくれてありがとう"])
def test_substantial_source_copy_is_detected_for_asian_scripts(source):
    assert is_source_copy(source + "!", source + ".")


@pytest.mark.parametrize(
    "source,target",
    [
        ("오늘은 여러분과 같이 이야기하고 싶어요", "今天，여러분과 같이 이야기하고 싶어요"),
        ("오늘은정말행복해요", "今天，오늘은정말행복해요"),
        ("오늘 Weverse에서 여러분과 같이 이야기하고 싶어요", "今天在Weverse，여러분과 같이 이야기하고 싶어요"),
    ],
)
def test_chinese_target_rejects_copied_korean_clause(source, target):
    assert is_untranslated_output(target, source, TargetLanguage.SIMPLIFIED_CHINESE, "ko")
    report = inspect_translation_batch(
        [SubtitleProcessData(1, source, source_language="ko")],
        [SubtitleProcessData(1, source, target, source_language="ko")],
        TargetLanguage.SIMPLIFIED_CHINESE,
    )
    assert report.untranslated == ["1"]


@pytest.mark.parametrize(
    "source,target",
    [
        ("네", "是的"), ("응", "嗯"), ("아니", "不是"), ("어?", "嗯？"),
        ("네 네 맞아요", "对，对，没错"),
        ("오늘 민지 봤어요", "今天见到 민지 了"),
        ("이루다 김민지 둘이 같이 했어요", "이루다和김민지一起做的"),
        ("이루다 김민지 둘이 같이 했어요", "是이루다 김민지一起做的"),
        ("투모로우 바이 투게더 좋아해요", "我喜欢 투모로우 바이 투게더"),
        ("오늘 Weverse live 해요", "今天在Weverse直播"),
        ("오늘 너무너무 행복해요", "今天特别特别开心"),
        ("삐용삐용 띠요", "呜哇呜哇，嘀呦"),
        ("오늘도 너를 사랑해요 틀어 주세요", "请放《오늘도 너를 사랑해요》"),
    ],
)
def test_short_replies_emphasis_names_and_mixed_language_are_allowed(source, target):
    assert not is_untranslated_output(target, source, TargetLanguage.SIMPLIFIED_CHINESE, "ko")


def test_korean_target_does_not_reject_its_own_script():
    assert not is_untranslated_output(
        "오늘은 여러분과 같이 이야기하고 싶어요", "I want to talk with you today.",
        TargetLanguage.KOREAN, "en",
    )


@pytest.mark.parametrize("language", ["en", "ja", "ko", "mixed"])
@pytest.mark.parametrize("source", ["BMW M3", "https://example.com", "123"])
def test_language_metadata_keeps_existing_neutral_identifiers_valid(language, source):
    assert not is_untranslated_output(
        source, source, TargetLanguage.SIMPLIFIED_CHINESE, language,
    )


def test_neutral_identifier_rule_does_not_exempt_korean_product_transliteration():
    # This needs separately confirmed term identity, not matching digits or
    # merely a product-looking target. Keep the known limitation explicit.
    assert is_untranslated_output(
        "iPhone18", "아이폰18", TargetLanguage.SIMPLIFIED_CHINESE, "ko",
    )
    assert is_untranslated_output(
        "Samsung18", "아이폰18", TargetLanguage.SIMPLIFIED_CHINESE, "ko",
    )


def test_japanese_paraphrased_emphasis_keeps_existing_runtime_behavior(translator):
    valid, _ = translator._validate_cross_key_boundaries(
        {"1": "我真的感到非常高兴", "2": "真的感到非常高兴呀"},
        {"1": "とても嬉しく思います", "2": "本当に幸せです"},
        str,
    )
    assert valid


@pytest.mark.parametrize(
    "source,target,expected",
    [
        ({"1": "오늘 정말 행복해요", "2": "오늘 정말 행복해요"},
         {"1": "今天真的非常开心", "2": "今天真的非常开心"}, True),
        ({"1": "너무 너무 좋아요", "2": "너무 좋아요"},
         {"1": "真的真的非常喜欢", "2": "真的非常喜欢"}, True),
        ({"1": "네", "2": "네"}, {"1": "对", "2": "对"}, True),
        ({"1": "삐용삐용 띠요", "2": "삐용삐용 띠요"},
         {"1": "呜哇呜哇嘀呦嘀呦", "2": "呜哇呜哇嘀呦嘀呦"}, True),
        ({"1": "오늘 저녁 먹었어요", "2": "오늘 연습하러 가요"},
         {"1": "今天已经吃过晚饭了", "2": "今天已经吃过晚饭了"}, False),
        ({"1": "오늘 정말 좋아요", "2": "오늘 정말 싫어요"},
         {"1": "今天真的非常喜欢", "2": "今天真的非常喜欢"}, False),
    ],
)
def test_runtime_repetition_uses_korean_source_evidence(translator, source, target, expected):
    valid, _ = translator._validate_cross_key_boundaries(target, source, str)
    assert valid is expected


def test_short_korean_repetition_guard_does_not_use_only_latin_words():
    assert LLMTranslator._is_disfluent_alignment_fragment("네 네 네")
    assert not LLMTranslator._is_disfluent_alignment_fragment("오늘 라이브 정말 재미있었어요")
    assert LLMTranslator._is_disfluent_alignment_fragment("of that of that struggle")


def test_korean_conversation_guidance_is_selected_only_for_korean():
    korean = target_language_style_rules("简体中文", ["네, 오늘 너무 좋아요"])
    assert "Preserve short Korean replies and reactions" in korean
    assert "do not default every subjectless statement to I" in korean
    assert "rather than an empty value" in korean
    for source in ["Yeah, I like it.", "はい、好きです。"]:
        assert "For informal Korean conversation" not in target_language_style_rules("简体中文", [source])
