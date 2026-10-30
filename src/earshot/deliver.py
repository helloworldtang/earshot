"""投递层:exomind API(直连,免 CLI)/ exomind CLI / webhook / 不投递,可插拔。"""

from __future__ import annotations

import hashlib
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from earshot.config import Settings

EXOMIND_ORIGIN = "earshot"
EXOMIND_PENDING_EXIT_CODE = 3  # exomind CLI 契约:3 = 已提交、后台仍在抽取,不是失败
DELIVER_TIMEOUT_S = 600.0
INGEST_POLL_INTERVAL_S = 2.0
INGEST_POLL_BUDGET_S = 120.0


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """投递结果。

    Attributes:
        status: ``delivered`` / ``pending`` / ``skipped`` / ``failed``。
        detail: 给用户看的一句话说明。
    """

    status: str
    detail: str


def build_exomind_args(
    md_path: Path,
    title: str,
    tags: list[str],
    origin: str = EXOMIND_ORIGIN,
) -> list[str]:
    """组装 exomind ingest 命令行(纯函数,wire 契约由单测钉死)。"""
    args = ["exomind", "ingest", "--file", str(md_path), "--origin", origin]
    if title:
        args += ["--title", title]
    for tag in tags:
        args += ["--tag", tag]
    return args


def deliver_exomind(
    md_path: Path,
    title: str,
    tags: list[str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DeliveryResult:
    """经 exomind CLI 投递(复用其异步 job、轮询与 client_key 幂等)。

    Args:
        md_path: 笔记文件。
        title: 标题(透传 ``--title``)。
        tags: 标签列表。
        runner: subprocess 运行器(测试注入用)。

    Returns:
        DeliveryResult,``pending`` 对应 CLI exit 3(已提交未完成)。
    """
    cmd = build_exomind_args(md_path, title, tags)
    try:
        proc = runner(cmd, capture_output=True, text=True, timeout=DELIVER_TIMEOUT_S)
    except FileNotFoundError:
        return DeliveryResult("failed", "未找到 exomind CLI(npm i -g exomind,或改用 --deliver webhook/none)")
    except subprocess.TimeoutExpired:
        return DeliveryResult("failed", f"exomind CLI 超时({DELIVER_TIMEOUT_S:.0f}s)")
    out_lines = (proc.stdout or "").strip().splitlines()
    tail = out_lines[-1] if out_lines else ""
    if proc.returncode == 0:
        return DeliveryResult("delivered", tail or "已投递进知识库")
    if proc.returncode == EXOMIND_PENDING_EXIT_CODE:
        return DeliveryResult("pending", "已提交,后台抽取中(exomind 会幂等续查回执)")
    err = (proc.stderr or "").strip().splitlines()
    detail = err[-1] if err else tail or f"exomind CLI 退出码 {proc.returncode}"
    return DeliveryResult("failed", detail)


def deliver_webhook(
    url: str,
    title: str,
    content_md: str,
    tags: list[str],
    origin: str = EXOMIND_ORIGIN,
    poster: Callable[[str, dict[str, str | list[str]]], None] | None = None,
) -> DeliveryResult:
    """POST JSON 到任意 webhook,适配非 exomind 知识库。

    Args:
        url: webhook 地址。
        title: 标题。
        content_md: Markdown 正文。
        tags: 标签。
        origin: 来源标识。
        poster: 发送函数(测试注入用),抛异常即视为失败。

    Returns:
        DeliveryResult。
    """

    def _post(target: str, payload: dict[str, str | list[str]]) -> None:
        resp = httpx.post(target, json=payload, timeout=30.0)
        resp.raise_for_status()

    send = poster or _post
    payload: dict[str, str | list[str]] = {"origin": origin, "title": title, "content": content_md, "tags": tags}
    try:
        send(url, payload)
    except Exception as e:  # httpx 异常族不稳定,统一收口
        return DeliveryResult("failed", f"webhook 投递失败: {e}")
    return DeliveryResult("delivered", f"已 POST 到 {url}")


def client_key_for(content: str) -> str:
    """内容幂等键:与服务端两层幂等(优先级最高)对齐,重复投递不会入库两次。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def deliver_exomind_api(
    md_path: Path,
    title: str,
    tags: list[str],
    settings: Settings,
    poster: Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]] | None = None,
    getter: Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]] | None = None,
    poll_interval_s: float = INGEST_POLL_INTERVAL_S,
    poll_budget_s: float = INGEST_POLL_BUDGET_S,
) -> DeliveryResult:
    """直连 exomind HTTP API 投递(免装 exomind CLI)。

    走 POST /ingest/async + GET /ingest/status,与 CLI 同契约:
    秒回 job_id → 轮询至 done/failed;超预算视为 pending(内容已入库,抽取仍在跑)。

    Args:
        md_path: 笔记文件。
        title: 标题。
        tags: 标签。
        settings: 含 base_url / api_key 的配置。
        poster: ``(url, payload, headers) -> 响应 dict``(测试注入)。
        getter: ``(url, params, headers) -> 响应 dict``(测试注入)。
        poll_interval_s / poll_budget_s: 轮询节奏与总预算。

    Returns:
        DeliveryResult。
    """
    if not settings.exomind_api_key:
        return DeliveryResult("failed", "无 exomind 凭据(调用方应回退 CLI)")

    def _post(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        return dict(resp.json())

    def _get(url: str, params: dict[str, str], headers: dict[str, str]) -> dict[str, Any]:
        resp = httpx.get(url, params=params, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return dict(resp.json())

    post = poster or _post
    get = getter or _get
    base = settings.exomind_base_url
    headers = {"Authorization": f"Bearer {settings.exomind_api_key}"}

    content = md_path.read_text(encoding="utf-8")
    payload: dict[str, Any] = {"content": content, "origin": EXOMIND_ORIGIN, "client_key": client_key_for(content)}
    if title:
        payload["title"] = title
    if tags:
        payload["tags"] = tags
    try:
        resp = post(f"{base}/ingest/async", payload, headers)
    except Exception as e:
        return DeliveryResult("failed", f"提交 ingest 失败: {e}")
    job_id = resp.get("job_id")
    if not job_id:
        return DeliveryResult("failed", f"响应缺 job_id: {str(resp)[:200]}")

    deadline = time.monotonic() + poll_budget_s
    while time.monotonic() < deadline:
        try:
            st = get(f"{base}/ingest/status", {"job_id": str(job_id)}, headers)
        except Exception as e:
            return DeliveryResult("failed", f"查询回执失败(job_id={job_id}): {e}")
        status = st.get("status")
        if status == "done":
            entities = st.get("entities") or []
            pages = st.get("created_pages") or []
            updated = st.get("updated_pages") or []
            detail = f"已入库:实体 {len(entities)} 个,页面 +{len(pages)}/更{len(updated)}(job_id={job_id})"
            return DeliveryResult("delivered", detail)
        if status == "failed":
            return DeliveryResult("failed", f"服务端抽取失败: {st.get('error', '未知')}(job_id={job_id})")
        time.sleep(poll_interval_s)
    return DeliveryResult("pending", f"已提交,后台抽取中(job_id={job_id})")
