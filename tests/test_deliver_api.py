"""deliver_exomind_api 单测:直连 HTTP 投递(wire 契约 + 轮询状态机 + 超长分片)。"""

from pathlib import Path
from typing import Any

from earshot.config import Settings
from earshot.deliver import (
    INGEST_MAX_CONTENT_CHARS,
    client_key_for,
    deliver_exomind_api,
    split_for_ingest,
)


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


def test_deliver_api_passes_source_url(tmp_path: Path) -> None:
    """wire 契约:video 路线带 source_url → payload['url'](服务端仅留痕不抓取)。"""
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")
    seen: dict[str, Any] = {}

    def poster(_u: str, payload: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        seen.update(payload)
        return {"job_id": "j-url"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done", "entities": 1}

    result = deliver_exomind_api(
        md,
        "t",
        [],
        _settings(),
        poster=poster,
        getter=getter,
        poll_interval_s=0,
        source_url="https://www.bilibili.com/video/BV16qpF64E6V/",
    )
    assert result.status == "delivered"
    assert seen["url"] == "https://www.bilibili.com/video/BV16qpF64E6V/"
    assert seen["content"]  # 正文照常在


def test_deliver_api_omits_url_when_absent(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")
    seen: dict[str, Any] = {}

    def poster(_u: str, payload: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        seen.update(payload)
        return {"job_id": "j2"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done"}

    deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert "url" not in seen  # listen/audio 路线不带


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


def test_split_for_ingest_short_content_untouched() -> None:
    assert split_for_ingest("短内容") == ["短内容"]
    assert split_for_ingest("x" * INGEST_MAX_CONTENT_CHARS) == ["x" * INGEST_MAX_CONTENT_CHARS]


def test_split_for_ingest_paragraph_greedy_no_loss() -> None:
    """质量点:各片均不超限,且不丢段、不重段、顺序不变。"""
    paras = [f"第{i}段" + "字" * 9000 for i in range(10)]  # 10 段×~9K = 90K → 4 片左右
    content = "\n\n".join(paras)
    chunks = split_for_ingest(content)
    assert len(chunks) > 1
    assert all(len(c) <= INGEST_MAX_CONTENT_CHARS for c in chunks)
    rejoined = [c for chunk in chunks for c in chunk.split("\n\n")]
    assert rejoined == paras


def test_split_for_ingest_single_giant_paragraph_hard_cut() -> None:
    """无段落边界(整篇一段)时按字符硬切,不丢字。"""
    content = "字" * (INGEST_MAX_CONTENT_CHARS + 5000)
    chunks = split_for_ingest(content)
    assert all(len(c) <= INGEST_MAX_CONTENT_CHARS for c in chunks)
    assert "".join(chunks) == content


def test_deliver_api_splits_long_content(tmp_path: Path) -> None:
    """质量点:超长转写自动分片逐片投递,标题带 (i/n),各片独立 client_key。"""
    content = "\n\n".join(f"第{i}段" + "字" * 9000 for i in range(6))  # ~54K → 2 片
    md = tmp_path / "note.md"
    md.write_text(content, encoding="utf-8")
    posts: list[dict[str, Any]] = []

    def poster(_u: str, payload: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        posts.append(payload)
        return {"job_id": f"j-{len(posts)}"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done", "entities": 2}

    result = deliver_exomind_api(md, "长转写", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "delivered"
    assert len(posts) == 2
    assert [p["title"] for p in posts] == ["长转写(1/2)", "长转写(2/2)"]
    assert len({p["client_key"] for p in posts}) == 2
    assert posts[0]["client_key"] == client_key_for(posts[0]["content"])
    assert sum(len(p["content"]) for p in posts) >= len(content) - 10  # 分隔符损耗
    assert "分 2 片" in result.detail


def test_deliver_api_chunk_failure_reports_partial(tmp_path: Path) -> None:
    """质量点:某片失败 → failed 并注明已成功片数,已入库片幂等可重跑。"""
    content = "\n\n".join(f"第{i}段" + "字" * 9000 for i in range(6))
    md = tmp_path / "note.md"
    md.write_text(content, encoding="utf-8")
    posts: list[dict[str, Any]] = []

    def poster(_u: str, payload: dict[str, Any], _h: dict[str, str]) -> dict[str, Any]:
        posts.append(payload)
        if len(posts) == 2:
            raise RuntimeError("HTTP 422: 内容太长")
        return {"job_id": f"j-{len(posts)}"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done", "entities": 1}

    result = deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert result.status == "failed"
    assert "已成功 1/2 片" in result.detail
    assert "HTTP 422" in result.detail  # 服务端 detail 必须透出,不吞成裸状态码


def test_deliver_api_post_error_surfaces_server_detail(tmp_path: Path, monkeypatch: Any) -> None:
    """质量点:4xx 响应体里的服务端 detail(如「内容太长」)必须进失败信息。"""

    class _FakeResp:
        is_error = True
        status_code = 422
        text = '{"detail":"内容太长"}'

        def json(self) -> dict[str, Any]:
            return {"detail": "内容太长(45570 字符),单次最多 30000 字符"}

    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResp(), raising=True)
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done"}

    result = deliver_exomind_api(md, "t", [], _settings(), getter=getter, poll_interval_s=0)
    assert result.status == "failed"
    assert "HTTP 422" in result.detail
    assert "45570" in result.detail
