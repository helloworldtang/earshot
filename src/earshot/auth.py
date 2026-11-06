"""设备流登录:复用 exomind 服务端 /auth/device/* 端点,写入 ~/.exomind/config.json。

earshot 单独安装(未装 emcli)时的取凭据出口;与 emcli 共写同一配置文件,单一凭据源不变。
智能登录:已有凭据探活仍有效则直接返回,不弹浏览器(--force 跳过)。
"""

from __future__ import annotations

import contextlib
import json
import socket
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from earshot.config import EXOMIND_CONFIG_FILENAME

Poster = Callable[[str, dict[str, Any]], dict[str, Any]]
Getter = Callable[[str, dict[str, str]], int]
Opener = Callable[[str], bool]
Sleeper = Callable[[float], None]


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(url, json=payload, timeout=30.0)
    resp.raise_for_status()
    return dict(resp.json())


def _get_status(url: str, headers: dict[str, str]) -> int:
    return int(httpx.get(url, headers=headers, timeout=30.0).status_code)


def _open_browser(url: str) -> bool:
    return webbrowser.open(url)


def verify_key(base_url: str, api_key: str, getter: Getter | None = None) -> bool:
    """探活已有凭据:打一次受保护端点 /keywords,200 即仍有效。

    Args:
        base_url: exomind API 地址。
        api_key: 待验证的凭据。
        getter: ``(url, headers) -> HTTP 状态码``,测试注入;缺省走 httpx。

    Returns:
        是否仍有效;401/403 与网络错误均返回 False(交给设备流给出更准确的错误)。
    """
    get_status = getter or _get_status
    try:
        return get_status(f"{base_url.rstrip('/')}/keywords", {"Authorization": f"Bearer {api_key}"}) == 200
    except httpx.HTTPError:
        return False


def device_login(
    base_url: str,
    poster: Poster | None = None,
    opener: Opener | None = None,
    sleeper: Sleeper | None = None,
) -> str:
    """设备流登录:发起 → 打开浏览器授权页 → 轮询拿 api_key。

    Args:
        base_url: exomind API 地址。
        poster: ``(url, payload) -> 响应 dict``,测试注入;缺省走 httpx。
        opener: 打开授权页的回调(返回是否成功),测试注入;缺省 webbrowser。
        sleeper: 轮询间隔等待,测试注入;缺省 time.sleep。

    Returns:
        服务端签发的 api_key(长期有效,写入 ~/.exomind/config.json 后投递/云转写均可用)。

    Raises:
        RuntimeError: 授权被拒、会话过期或等待超时。
        httpx.HTTPError: 网络或服务端错误(5xx 等)。
    """
    post = poster or _post_json
    open_page = opener or _open_browser
    sleep = sleeper or time.sleep

    info = post(
        f"{base_url.rstrip('/')}/auth/device/code",
        {"name": f"earshot@{socket.gethostname()[:60]}"},
    )
    device_code = str(info["device_code"])
    user_code = str(info.get("user_code", ""))
    interval = max(1, int(info.get("interval", 3)))
    expires_in = max(30, int(info.get("expires_in", 600)))
    url = str(info.get("verification_url") or "")
    if not url.startswith("http"):
        url = f"{base_url.rstrip('/')}/ui/device?code={user_code}"

    print(f"  1. 打开 {url}")
    print(f"  2. 确认设备码 {user_code},点击「授权登录」")
    open_page(url)

    deadline = time.monotonic() + expires_in
    while time.monotonic() < deadline:
        sleep(interval)
        try:
            result = post(f"{base_url.rstrip('/')}/auth/device/token", {"device_code": device_code})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400:
                raise
            detail = ""
            with contextlib.suppress(ValueError):
                detail = str((exc.response.json() or {}).get("error", ""))
            if detail == "authorization_pending":
                continue
            raise RuntimeError("设备登录会话已过期,请重新执行 earshot login") from exc
        api_key = str(result.get("api_key") or "")
        if api_key:
            return api_key
    raise RuntimeError(f"等待授权超时({expires_in}s),请重新执行 earshot login")


def save_credentials(
    base_url: str,
    api_key: str,
    config_path: Path | None = None,
) -> Path:
    """写入凭据到 ~/.exomind/config.json(0600),保留既有其他字段。

    Args:
        base_url: exomind API 地址。
        api_key: 登录拿到的凭据。
        config_path: 配置文件路径,测试注入;缺省 ``~/.exomind/config.json``。

    Returns:
        实际写入的配置文件路径。
    """
    path = config_path if config_path is not None else Path.home() / EXOMIND_CONFIG_FILENAME
    data: dict[str, str] = {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = {k: v for k, v in raw.items() if isinstance(v, str)}
    except (OSError, ValueError):
        pass
    data["base_url"] = base_url.rstrip("/")
    data["api_key"] = api_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path
