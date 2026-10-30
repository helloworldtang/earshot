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


def ffmpeg_exe() -> str | None:
    """定位 ffmpeg:优先系统 PATH,其次 imageio-ffmpeg wheel 内置二进制。

    Returns:
        ffmpeg 可执行文件路径,两级都无则 None。
    """
    found = shutil.which(FFMPEG)
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415 延迟导入:装了系统 ffmpeg 就不必加载

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # imageio_ffmpeg 缺失或平台无二进制
        return None


def ffmpeg_available() -> bool:
    """ffmpeg 是否可用(系统或内置 wheel 任一)。"""
    return ffmpeg_exe() is not None


def to_wav16k(src: Path, dst: Path) -> None:
    """抽音轨并重采样为 16k mono wav。

    Args:
        src: 输入任意音视频文件(ffmpeg 支持即行)。
        dst: 输出 wav 路径。

    Raises:
        ExtractError: ffmpeg 缺失或执行失败。
    """
    exe = ffmpeg_exe()
    if exe is None:
        msg = "未找到 ffmpeg(系统 PATH 与 imageio-ffmpeg 内置均无;重装 earshot 依赖)"
        raise ExtractError(msg)
    cmd = [
        exe,
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
