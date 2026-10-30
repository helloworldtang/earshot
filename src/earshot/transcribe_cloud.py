"""云端转写:复用 exomind 服务端 /transcribe/upload + /transcribe/status(dashscope)。"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from earshot.config import Settings

POLL_INTERVAL_S = 3.0
DEFAULT_TIMEOUT_S = 900.0
UPLOAD_TIMEOUT_S = 120.0


class CloudTranscribeError(RuntimeError):
    """云端转写失败(未配置凭据 / 上传失败 / 转写失败 / 超时)。"""


def transcribe_cloud(wav: Path, settings: Settings, timeout_s: float = DEFAULT_TIMEOUT_S) -> str:
    """上传 wav 并轮询至转写完成。

    Args:
        wav: 音频文件(16k wav;服务端另收 .aac/.mp3/.m4a)。
        settings: 含 exomind_base_url / api_key 的配置。
        timeout_s: 轮询总预算。

    Returns:
        转写全文。

    Raises:
        CloudTranscribeError: 任一环节失败。
    """
    if not settings.exomind_api_key:
        msg = "未配置 exomind 凭据(先 `exomind login` 生成 ~/.exomind/config.json,或设 EXOMIND_API_KEY)"
        raise CloudTranscribeError(msg)
    base = settings.exomind_base_url
    headers = {"Authorization": f"Bearer {settings.exomind_api_key}"}

    try:
        with wav.open("rb") as f:
            resp = httpx.post(
                f"{base}/transcribe/upload",
                files={"file": (wav.name, f, "audio/wav")},
                headers=headers,
                timeout=UPLOAD_TIMEOUT_S,
            )
    except httpx.HTTPError as e:
        msg = f"上传失败: {e}"
        raise CloudTranscribeError(msg) from e
    if resp.status_code != 200:
        msg = f"上传被拒({resp.status_code}): {resp.text[:300]}"
        raise CloudTranscribeError(msg)
    task_id = resp.json().get("task_id")
    if not task_id:
        msg = f"响应缺 task_id: {resp.text[:300]}"
        raise CloudTranscribeError(msg)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/transcribe/status", params={"task_id": task_id}, headers=headers, timeout=30.0)
        except httpx.HTTPError as e:
            msg = f"轮询失败: {e}"
            raise CloudTranscribeError(msg) from e
        payload = r.json()
        status = payload.get("status")
        if status == "done":
            return str(payload.get("text") or "")
        if status == "failed":
            msg = f"转写失败: {payload.get('error', '未知错误')}"
            raise CloudTranscribeError(msg)
        time.sleep(POLL_INTERVAL_S)
    msg = f"转写超时({timeout_s:.0f}s),task_id={task_id}"
    raise CloudTranscribeError(msg)
