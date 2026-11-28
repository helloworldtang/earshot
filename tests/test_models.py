"""模型清单单测:核心下载集与实时粗稿集分离,批量路线不背 190MB 粗稿模型。"""

from pathlib import Path

from earshot.models import MODEL_FILES, STREAM_MODEL_FILES, missing_models, missing_stream_models


def test_stream_models_not_in_core_set() -> None:
    """audio/批量路线的 ensure_models 默认不下粗稿模型:核心集不含 asr_stream。"""
    assert all(not rel.startswith("asr_stream/") for rel in MODEL_FILES)
    assert all(rel.startswith("asr_stream/") for rel in STREAM_MODEL_FILES)


def test_missing_models_distinguish_core_and_stream(tmp_path: Path) -> None:
    assert len(missing_models(tmp_path)) == len(MODEL_FILES)
    assert len(missing_stream_models(tmp_path)) == len(STREAM_MODEL_FILES)
    (tmp_path / "asr_stream").mkdir()
    (tmp_path / "asr_stream" / "tokens.txt").touch()
    assert "asr_stream/tokens.txt" not in missing_stream_models(tmp_path)
    assert len(missing_stream_models(tmp_path)) == len(STREAM_MODEL_FILES) - 1
