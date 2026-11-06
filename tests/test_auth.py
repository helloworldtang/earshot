"""设备流登录测试:全 mock 网络与浏览器,无真实请求。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from earshot.auth import device_login, save_credentials, verify_key


def test_verify_key_three_branches() -> None:
    """探活:200→True;401→False;网络错误→False(交给设备流)。"""
    assert verify_key("https://x.test/", "sk_ok", getter=lambda url, headers: 200) is True
    assert verify_key("https://x.test", "sk_bad", getter=lambda url, headers: 401) is False

    def _raise(url: str, headers: dict[str, str]) -> int:
        raise httpx.ConnectError("network down")

    assert verify_key("https://x.test", "sk_x", getter=_raise) is False


def test_device_login_pending_then_key() -> None:
    """发起→pending 一次→拿到 key;verification_url 相对路径拼成绝对;浏览器被打开。"""
    calls: list[str] = []
    opened: list[str] = []

    def poster(url: str, payload: dict[str, object]) -> dict[str, object]:
        calls.append(url)
        if url.endswith("/auth/device/code"):
            return {
                "device_code": "dc-1",
                "user_code": "AB12-CD34",
                "expires_in": 60,
                "interval": 1,
                "verification_url": "/ui/device",
            }
        if url.endswith("/auth/device/token"):
            if len(calls) == 2:
                raise httpx.HTTPStatusError(
                    "pending",
                    request=httpx.Request("POST", url),
                    response=httpx.Response(400, json={"error": "authorization_pending"}),
                )
            return {"api_key": "sk_dev_1"}
        raise AssertionError(f"意外请求: {url}")

    def opener(url: str) -> bool:
        opened.append(url)
        return True

    key = device_login(
        "https://x.test",
        poster=poster,
        opener=opener,
        sleeper=lambda s: None,
    )
    assert key == "sk_dev_1"
    assert opened and "code=AB12-CD34" in opened[0], "相对 verification_url 应拼成带设备码的绝对地址"


def test_device_login_expired_session() -> None:
    """400 非 pending(expires)→ 明确报错会话过期。"""

    def poster(url: str, payload: dict[str, object]) -> dict[str, object]:
        if url.endswith("/auth/device/code"):
            return {"device_code": "dc-x", "user_code": "AAAA-BBBB", "expires_in": 60, "interval": 1}
        raise httpx.HTTPStatusError(
            "expired", request=httpx.Request("POST", url), response=httpx.Response(400, json={"error": "expired_token"})
        )

    with pytest.raises(RuntimeError, match="过期"):
        device_login("https://x.test", poster=poster, opener=lambda url: True, sleeper=lambda s: None)


def test_save_credentials_merges_and_chmods(tmp_path: Path) -> None:
    """写入保留既有其他字段;权限 0600;损坏旧文件可覆盖。"""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"base_url": "https://old.example", "extra": "keep"}), encoding="utf-8")

    path = save_credentials("https://x.test/", "sk_new", config_path=cfg)
    assert path == cfg
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["base_url"] == "https://x.test", "去尾斜杠且更新"
    assert data["api_key"] == "sk_new"
    assert data["extra"] == "keep", "既有其他字段保留"
    assert cfg.stat().st_mode & 0o777 == 0o600

    cfg.write_text("not-json", encoding="utf-8")
    save_credentials("https://x.test", "sk_again", config_path=cfg)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["api_key"] == "sk_again", "损坏旧文件可覆盖"
