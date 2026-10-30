"""ffmpeg 工具:任意音视频 → 16k mono wav(ASR 统一入口格式)。"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

FFMPEG = "ffmpeg"
TARGET_SAMPLE_RATE = 16000


class ExtractError(RuntimeError):
    """ffmpeg 抽音轨失败。"""


def ffmpeg_available() -> bool:
    """ffmpeg 是否在 PATH 上。"""
    return shutil.which(FFMPEG) is not None


def to_wav16k(src: Path, dst: Path) -> None:
    """抽音轨并重采样为 16k mono wav。

    Args:
        src: 输入任意音视频文件(ffmpeg 支持即行)。
        dst: 输出 wav 路径。

    Raises:
        ExtractError: ffmpeg 缺失或执行失败。
    """
    if not ffmpeg_available():
        msg = "未找到 ffmpeg(macOS: brew install ffmpeg)"
        raise ExtractError(msg)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # noqa: S603
    if proc.returncode != 0:
        msg = f"ffmpeg 失败({proc.returncode}): {(proc.stderr or '').strip()[:500]}"
        raise ExtractError(msg)


def wav_duration(path: Path) -> float:
    """读 wav 时长(秒)。

    Raises:
        ExtractError: 文件不是合法 wav。
    """
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except wave.Error as e:
        msg = f"不是合法 wav: {path}({e})"
        raise ExtractError(msg) from e
