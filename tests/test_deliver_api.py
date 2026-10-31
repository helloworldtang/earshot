"""deliver_exomind_api 单测:直连 HTTP 投递(wire 契约 + 轮询状态机)。"""

from pathlib import Path
from typing import Any

from earshot.config import Settings
from earshot.deliver import client_key_for, deliver_exomind_api


def _settings() -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://api.example.com",
        exomind_api_key="key-1",
    )


def test_client_key_stable_and_content_bound() -> None:
    assert client_key_for("abc") == client_key_for("abc")
    assert client_key_for("abc") != client_key_for("abd")
    assert len(client_key_for("abc")) == 64


def test_deliver_api_done_receipt_counts_as_int(tmp_path: Path) -> None:
    """回执字段为个数(int)时也须正常 —— 真实事故:entities 为 int 导致 len() 崩溃。"""
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")

    def poster(_u: str, _p: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        return {"job_id": "j-int"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done", "entities": 7, "created_pages": 2, "updated_pages": 3}

    result = deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "delivered"
    assert "实体 7 个" in result.detail
    assert "+2/更3" in result.detail


def test_deliver_api_done_receipt(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# 标题\n正文", encoding="utf-8")
    seen: dict[str, Any] = {}

    def poster(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        seen["post_url"] = url
        seen["payload"] = payload
        seen["auth"] = headers.get("Authorization")
        return {"job_id": "j-123", "status": "pending"}

    def getter(url: str, params: dict[str, str], headers: dict[str, str]) -> dict[str, Any]:
        seen["get_url"] = url
        seen["params"] = params
        return {"status": "done", "entities": [{"name": "a"}, {"name": "b"}], "created_pages": ["p1"]}

    result = deliver_exomind_api(md, "标题", ["旁听"], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "delivered"
    assert "实体 2 个" in result.detail
    # wire 契约
    assert seen["post_url"] == "https://api.example.com/ingest/async"
    assert seen["auth"] == "Bearer key-1"
    payload = seen["payload"]
    assert payload["origin"] == "earshot"
    assert payload["title"] == "标题"
    assert payload["tags"] == ["旁听"]
    assert payload["client_key"] == client_key_for("# 标题\n正文")
    assert seen["get_url"] == "https://api.example.com/ingest/status"
    assert seen["params"] == {"job_id": "j-123"}


def test_deliver_api_pending_on_budget_timeout(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")

    def poster(_u: str, _p: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        return {"job_id": "j-9", "status": "pending"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "running"}

    result = deliver_exomind_api(
        md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0, poll_budget_s=0.01
    )
    assert result.status == "pending"
    assert "j-9" in result.detail


def test_deliver_api_failed_status(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")

    def poster(_u: str, _p: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        return {"job_id": "j-x"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "failed", "error": "quota"}

    result = deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "failed"
    assert "quota" in result.detail


def test_deliver_api_requires_key(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")
    s = Settings(
        home=Path("/tmp/x"), model_root=Path("/tmp/x/m"), exomind_base_url="https://api.example.com", exomind_api_key=""
    )
    result = deliver_exomind_api(md, "t", [], s, poster=lambda *_: {}, getter=lambda *_: {})
    assert result.status == "failed"


def test_missing_models_detection(tmp_path: Path) -> None:
    from earshot.models import MODEL_FILES, missing_models

    assert missing_models(tmp_path) == list(MODEL_FILES)
    (tmp_path / "asr").mkdir()
    (tmp_path / "asr" / "tokens.txt").write_text("x", encoding="utf-8")
    assert missing_models(tmp_path) == ["asr/model.int8.onnx", "vad/silero_vad.onnx"]


def test_deliver_api_ctrl_c_during_poll_is_pending(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")

    def poster(_u: str, _p: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        return {"job_id": "j-int"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        raise KeyboardInterrupt

    result = deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "pending"
    assert "不再等待回执" in result.detail
    assert "j-int" in result.detail
