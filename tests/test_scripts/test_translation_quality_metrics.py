import json
import unicodedata
from pathlib import Path

import pytest

from scripts.translation_quality.manifest import CorpusManifest, CorpusSample, sha256_file
from scripts.translation_quality.metrics import (
    METRICS_REVISION,
    _adjacent_duplicate_risks,
    _compact,
    _document_quality,
    _is_placeholder,
    _is_untranslated,
    _similarity,
    _target_language,
    evaluate_manifest,
)
from scripts.translation_quality.report import write_report
from scripts.translation_quality.srt import SrtCue, SrtDocument
from subforge.core.translate.types import TargetLanguage


def _write(path: Path, target_one: str, target_two: str) -> Path:
    path.write_text(
        f"""1
00:00:00,000 --> 00:00:01,000
{target_one}
First source.

2
00:00:01,000 --> 00:00:02,000
{target_two}
Second source.
""",
        encoding="utf-8",
    )
    return path


def test_evaluate_manifest_reports_human_changes_without_exposing_text(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
First source.

2
00:00:01,000 --> 00:00:02,000
Second source.
""",
        encoding="utf-8",
    )
    machine = _write(tmp_path / "machine.srt", "第一条机器译文", "第二条机器译文")
    gold = _write(tmp_path / "gold.srt", "第一条人工译文", "第二条机器译文")
    sample = CorpusSample(
        id="sample",
        title="Sample",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=2000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="deepseek-v4-flash",
        algorithm_version="test",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )
    manifest = CorpusManifest(1, "test", "now", "ROOT", (sample,))

    report = evaluate_manifest(manifest, tmp_path, manifest_hash="abc")

    assert report.aggregate["sample_count"] == 1
    assert report.aggregate["human_changed_cues"] == 1
    assert report.aggregate["hard_failure_count"] == 0
    assert report.samples[0].structure["exact"] is True

    output = tmp_path / "report"
    write_report(report, output)
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "第一条机器译文" not in markdown
    assert json.loads((output / "aggregate.json").read_text())["human_changed_cues"] == 1


def test_evaluate_manifest_counts_placeholders_and_reasoning_leaks(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        """1
00:00:00,000 --> 00:00:01,000
First source.

2
00:00:01,000 --> 00:00:02,000
Second source.
""",
        encoding="utf-8",
    )
    machine = _write(tmp_path / "machine.srt", "待翻译", "<think>分析</think>")
    gold = _write(tmp_path / "gold.srt", "第一条", "第二条")
    sample = CorpusSample(
        id="sample",
        title="Sample",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=2000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    report = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    )

    assert report.aggregate["placeholder_targets"] == 1
    assert report.aggregate["reasoning_leaks"] == 1
    assert report.aggregate["hard_failure_count"] == 2


def test_report_redacts_holdout_cue_indices(tmp_path: Path):
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nFirst source.\n", encoding="utf-8"
    )
    machine = tmp_path / "machine.srt"
    machine.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n待翻译\nFirst source.\n", encoding="utf-8"
    )
    gold = tmp_path / "gold.srt"
    gold.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n人工译文\nFirst source.\n", encoding="utf-8"
    )
    sample = CorpusSample(
        id="holdout",
        title="Holdout",
        split="holdout",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    payload = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    ).to_dict()

    holdout = payload["samples"][0]
    assert holdout["details_redacted"] is True
    assert "placeholder_targets" not in holdout["machine"]
    assert holdout["machine"]["placeholder_targets_count"] == 1
    assert "changed_cues" not in holdout["gold_comparison"]


def test_placeholder_detection_does_not_match_normal_words_containing_lue():
    assert _is_placeholder("（此句合并至上一句）") is True
    assert _is_placeholder("内容同上") is True
    assert _is_placeholder("比最终造价略便宜一些") is False
    assert _is_placeholder("这是更广泛旅游战略的一部分") is False


def test_reasoning_metric_does_not_treat_normal_result_narration_as_private_reasoning(
    tmp_path: Path,
):
    source = tmp_path / "source.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHere are the results.\n",
        encoding="utf-8",
    )
    machine = tmp_path / "machine.srt"
    machine.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n以下是结果\nHere are the results.\n",
        encoding="utf-8",
    )
    gold = tmp_path / "gold.srt"
    gold.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n结果如下\nHere are the results.\n",
        encoding="utf-8",
    )
    sample = CorpusSample(
        id="normal-result",
        title="Normal Result",
        split="development",
        domain="other",
        speaker_mode="unknown",
        speaker_count=None,
        source_languages=("en",),
        target_language="zh-CN",
        duration_ms=1000,
        source_srt=source.name,
        machine_srt=machine.name,
        gold_srt=gold.name,
        source_media=None,
        machine_model="unknown",
        algorithm_version="unknown",
        configuration={},
        provenance={},
        alignment={"cue_structure": "exact"},
        known_issues=(),
        notes="",
        hashes={
            "source_sha256": sha256_file(source),
            "machine_sha256": sha256_file(machine),
            "gold_sha256": sha256_file(gold),
        },
    )

    report = evaluate_manifest(
        CorpusManifest(1, "test", "now", "ROOT", (sample,)),
        tmp_path,
        manifest_hash="abc",
    )

    assert report.aggregate["reasoning_leaks"] == 0


def _cue(index: int, source: str, target: str) -> SrtCue:
    return SrtCue(index, "", index * 1000, (index + 1) * 1000, source, target, (target, source))


@pytest.mark.parametrize(
    "left,right",
    [
        ("오늘 저녁은 먹었어요", "새 앨범 준비하고 있어요"),
        ("今日はご飯を食べました", "新しい曲を練習しています"),
        ("I already had dinner.", "We are rehearsing a new song."),
    ],
)
def test_multilingual_metric_detects_copied_translation_for_different_sources(left, right):
    target = "今天晚上已经吃过晚饭了"
    risks = _adjacent_duplicate_risks((_cue(1, left, target), _cue(2, right, target)))
    assert len(risks) == 1
    assert risks[0]["source_similarity"] < 0.82


@pytest.mark.parametrize(
    "source,target",
    [
        ("정말 정말 고마워요", "真的真的非常感谢大家"),
        ("本当に本当にありがとう", "真的真的非常感谢大家"),
        ("Thank you so, so much.", "真的真的非常感谢大家"),
        ("네 네", "对对"),
        ("아니요", "不是"),
        ("응", "嗯"),
        ("はい", "是的"),
        ("Yes.", "对"),
    ],
)
def test_multilingual_metric_preserves_real_repetition_and_short_answers(source, target):
    assert not _adjacent_duplicate_risks((_cue(1, source, target), _cue(2, source, target)))
    assert not _is_untranslated(_cue(1, source, target))


@pytest.mark.parametrize("source", ["오늘은 집에 있어요", "今日は家にいます", "I am at home."])
def test_multilingual_metric_finds_untranslated_source(source):
    assert _is_untranslated(_cue(1, source, source))


@pytest.mark.parametrize("source", ["BMW M3", "https://example.com", "123"])
def test_untranslated_metric_preserves_neutral_identifiers(source):
    assert not _is_untranslated(_cue(1, source, source))


def test_metric_normalizes_decomposed_hangul_without_erasing_korean():
    source = "오늘 같이 이야기해요"
    assert _compact(source)
    assert _similarity(source, unicodedata.normalize("NFD", source)) == 1.0
    assert _similarity(source, "내일 새로운 노래를 들어요") < 0.82


def test_duplicate_metric_needs_source_evidence():
    target = "今天晚上已经吃过晚饭了"
    assert not _adjacent_duplicate_risks((_cue(1, "...", target), _cue(2, "안녕하세요", target)))


@pytest.mark.parametrize(
    "code,expected",
    [
        ("zh-CN", TargetLanguage.SIMPLIFIED_CHINESE),
        ("zh-Hans", TargetLanguage.SIMPLIFIED_CHINESE),
        ("zh_TW", TargetLanguage.TRADITIONAL_CHINESE),
        ("zh-Hant", TargetLanguage.TRADITIONAL_CHINESE),
        ("ja", TargetLanguage.JAPANESE),
        ("ko", TargetLanguage.KOREAN),
        ("en", TargetLanguage.ENGLISH),
        ("unknown", None),
    ],
)
def test_metric_respects_declared_target_language(code, expected):
    assert _target_language(code) == expected


def test_japanese_target_is_not_misreported_as_missing_chinese():
    document = SrtDocument(
        Path("synthetic.srt"),
        (_cue(1, "오늘은 집에 있어요", "今日は家にいます"),),
        "utf-8", "lf", False,
    )
    assert not _document_quality(document, TargetLanguage.JAPANESE, "ko")["untranslated_targets"]


@pytest.mark.parametrize("language", ["en", "mixed"])
def test_document_metric_with_declared_language_preserves_neutral_identifiers(language):
    document = SrtDocument(
        Path("synthetic.srt"),
        tuple(_cue(index, source, source) for index, source in enumerate(
            ["BMW M3", "https://example.com", "123"], 1,
        )),
        "utf-8", "lf", False,
    )
    assert not _document_quality(
        document, TargetLanguage.SIMPLIFIED_CHINESE, language,
    )["untranslated_targets"]


def test_evaluation_reports_record_detector_revision():
    report = evaluate_manifest(CorpusManifest(1, "empty", "now", "ROOT", ()), Path("."), manifest_hash="x")
    assert report.to_dict()["metrics_revision"] == METRICS_REVISION
