from argparse import Namespace
from unittest.mock import Mock

import pytest

from subforge.cli import exit_codes as EXIT
from subforge.cli.commands import transcribe
from subforge.core import asr
from subforge.core.asr.asr_data import ASRData, ASRDataSeg


@pytest.mark.parametrize("reason", ["context_disagreement", "final_export_gap", "final_audit_unavailable"])
def test_transcribe_preserves_partial_result_and_stops_progress(tmp_path, monkeypatch, reason):
    source = tmp_path / "source.wav"
    source.touch()
    destination = tmp_path / "source.srt"
    destination.write_text("existing subtitle", encoding="utf-8")
    data = ASRData([ASRDataSeg("Confirmed speech.", 0, 1000)])
    data.coverage_issues = [
        {"start": 0, "end": 1, "reason": "suspected_repetition"},
        {"start": 10, "end": 14.2, "reason": reason},
    ]
    monkeypatch.setattr(asr, "transcribe", lambda *args, **kwargs: data)
    progress = Mock()
    progress.start.return_value = progress
    monkeypatch.setattr(transcribe.output, "ProgressLine", lambda message: progress)
    args = Namespace(input=str(source), output=str(destination), quiet=False)
    status = transcribe.run(args, {"transcribe": {"asr": "whisperx"}})
    assert status == EXIT.RUNTIME_ERROR
    assert destination.read_text() == "existing subtitle"
    assert "Confirmed speech." in (tmp_path / "source_recovery.srt").read_text(encoding="utf-8-sig")
    progress.fail.assert_called_once()
    progress.finish.assert_not_called()


@pytest.mark.parametrize("reason", ["suspected_non_speech_text", "suspected_repetition"])
@pytest.mark.parametrize("quiet", [False, True])
def test_review_only_transcription_completes_without_losing_text(tmp_path, monkeypatch, capsys, reason, quiet):
    source = tmp_path / "source.wav"
    source.touch()
    destination = tmp_path / "source.srt"
    original = "사랑해 " * 20
    data = ASRData([ASRDataSeg(original, 0, 15000)])
    data.coverage_issues = [{"start": 0, "end": 15, "reason": reason}]
    monkeypatch.setattr(asr, "transcribe", lambda *args, **kwargs: data)
    progress = Mock()
    progress.start.return_value = progress
    monkeypatch.setattr(transcribe.output, "ProgressLine", lambda message: progress)

    status = transcribe.run(
        Namespace(input=str(source), output=str(destination), quiet=quiet),
        {"transcribe": {"asr": "whisperx"}},
    )

    assert status == EXIT.SUCCESS
    assert original.strip() in destination.read_text(encoding="utf-8-sig")
    assert not (tmp_path / "source_recovery.srt").exists()
    progress.fail.assert_not_called()
    assert progress.finish.call_count == (0 if quiet else 1)
    captured = capsys.readouterr()
    assert "need review against the audio" in captured.err
    assert "translation may continue" in captured.err
    if quiet:
        assert captured.out.strip() == str(destination)
