"""投递层:exomind CLI / webhook / 不投递,三种可插拔。"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

EXOMIND_ORIGIN = "earshot"
EXOMIND_PENDING_EXIT_CODE = 3  # exomind CLI 契约:3 = 已提交、后台仍在抽取,不是失败
DELIVER_TIMEOUT_S = 600.0


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
