"""显式 UA 护栏(可观测性标配,2026-11-09):自有服务(exomind)流量必须带
earshot/<version>,默认 python-httpx/x 即未收编。

质量点:①ua_headers 单一注入点格式与合并;②deliver_exomind_api 的 wire 头
真实携带(经注入 poster 捕获);③webhook(第三方)不经 ua_headers——无自有
服务端可分桶,不乱标。
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from earshot import __version__, ua_headers
from earshot.config import Settings
from earshot.deliver import deliver_exomind_api, deliver_webhook


def _settings() -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://api.example.com",
        exomind_api_key="key-1",
    )


def test_ua_headers_format_and_merge() -> None:
    h = ua_headers(Authorization="Bearer k")
    assert h["User-Agent"] == f"earshot/{__version__}"
    assert h["Authorization"] == "Bearer k"
    assert __version__ not in ("", "0.0.dev0"), "安装态下版本必须来自包元数据,不再是漂移的手写常量"


def test_deliver_api_carries_ua_on_wire(tmp_path: Path) -> None:
    md = tmp_path / "n.md"
    md.write_text("内容", encoding="utf-8")
    seen: dict[str, str] = {}

    def poster(_u: str, _p: dict[str, Any], h: dict[str, str]) -> dict[str, Any]:
        seen.update(h)
        return {"job_id": "j"}

    def getter(_u: str, _p: dict[str, str], _h: dict[str, str]) -> dict[str, Any]:
        return {"status": "done", "entities": 1}

    deliver_exomind_api(md, "t", [], _settings(), poster=poster, getter=getter, poll_interval_s=0)
    assert seen["User-Agent"].startswith("earshot/")
    assert seen["Authorization"].startswith("Bearer ")


def test_webhook_does_not_claim_earshot_ua(monkeypatch: pytest.MonkeyPatch) -> None:
    """第三方 webhook 不打 earshot UA——收编范围=自有服务,乱标反而污染对方统计。"""
    sent: list[Any] = []

    def fake_post(url: str, **kw: Any) -> httpx.Response:
        sent.append(kw.get("headers"))  # None = 未传头
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "post", fake_post)
    deliver_webhook("https://example.org/hook", "t", "c", ["a"])
    assert sent == [None]
