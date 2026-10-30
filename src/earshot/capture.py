"""系统声环回采集(soundcard):平台差异只允许存在于此模块。

- Windows: WASAPI loopback,免驱
- Linux: PulseAudio ``.monitor``,免驱
- macOS: 无原生环回,需安装 BlackHole 并(建议用多输出设备)设为输出
"""

from __future__ import annotations

import platform
import warnings
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# soundcard 在 macOS import 时必发一条"不支持环回"警告,属已知事实(BlackHole 方案的前提),
# 压掉以免吓到用户 —— 必须在 import soundcard 之前过滤
warnings.filterwarnings("ignore", message=".*loopback recording functionality.*", category=Warning)
import soundcard as sc  # noqa: E402

SAMPLE_RATE = 48000  # 环回设备主流采样率;重采样交给 ffmpeg
BLOCK_FRAMES = 4800  # 100ms/块,兼顾电平刷新频率与 CPU
BLACKHOLE_HINT = "blackhole"


def list_capture_devices() -> list[tuple[str, bool]]:
    """列出全部输入设备。

    Returns:
        ``(设备名, 是否环回)`` 列表。
    """
    return [(m.name, bool(m.isloopback)) for m in sc.all_microphones(include_loopback=True)]


def blackhole_missing_hint() -> str | None:
    """macOS 上未发现 BlackHole 设备时的安装指引,其他平台返回 None。"""
    if platform.system() != "Darwin":
        return None
    names = " ".join(name for name, _ in list_capture_devices()).lower()
    if BLACKHOLE_HINT in names:
        return None
    return (
        "未检测到 BlackHole。安装: brew install blackhole-2ch;"
        "再在「音频 MIDI 设置」建多输出设备(BlackHole+扬声器)并设为系统输出,"
        "即可边听边录。"
    )


def pick_loopback(hint: str | None = None) -> Any:
    """选采集设备:默认取当前默认扬声器的同名环回/输入设备。

    Args:
        hint: 设备名子串,大小写不敏感;None 时走默认逻辑。

    Returns:
        soundcard 的 Mic 对象。

    Raises:
        LookupError: 找不到匹配设备。
    """
    mics = sc.all_microphones(include_loopback=True)
    if hint:
        low = hint.lower()
        for m in mics:
            if low in m.name.lower():
                return m
        msg = f"未找到匹配「{hint}」的输入设备(earshot devices 查看)"
        raise LookupError(msg)
    speaker = sc.default_speaker()
    for m in mics:
        if m.name == speaker.name:  # macOS:BlackHole 输出即同名输入;Win/Linux:同名环回
            return m
    msg = f"默认扬声器「{speaker.name}」没有可采集的环回设备。{blackhole_missing_hint() or ''}"
    raise LookupError(msg)


def pick_device(hint: str | None = None) -> tuple[Any, str]:
    """选采集设备,带麦录兜底(零门槛优先于零音质损失)。

    优先级:hint 匹配 > 默认扬声器环回 > 任意环回 > 默认麦克风(外放收音,豆包同款)。

    Args:
        hint: 设备名子串;None 走默认回退链。

    Returns:
        ``(mic, mode_label)``,mode_label 用于向用户说明当前采集方式。

    Raises:
        LookupError: 全链条找不到可用设备。
    """
    try:
        return pick_loopback(hint), "环回内录(音质无损)"
    except LookupError:
        if hint:
            raise  # 显式指定的设备找不到,不擅自换
    for m in sc.all_microphones(include_loopback=True):
        if m.isloopback:
            return m, f"环回内录(音质无损,设备:{m.name})"
    mic = sc.default_microphone()
    if mic is None:
        msg = "未找到任何可用输入设备(earshot devices 查看)"
        raise LookupError(msg)
    fallback = f"麦克风收音(外放收音,豆包同款;配 BlackHole 可内录)设备:{mic.name}"
    return mic, fallback


def record_to_wav(mic: Any, out_wav: Path, on_level: Callable[[float], None] | None = None) -> float:
    """录制到 wav(pcm16/48k/mono),Ctrl-C 停止。

    Args:
        mic: pick_device 返回的设备。
        out_wav: 输出路径。
        on_level: 可选电平回调(rms,0~1 区间参考),用于终端电平条。

    Returns:
        实际录制时长(秒)。
    """
    total_frames = 0
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        try:
            with mic.recorder(samplerate=SAMPLE_RATE) as rec:
                while True:
                    data = rec.record(numframes=BLOCK_FRAMES)
                    mono = data.mean(axis=1) if data.ndim > 1 else np.asarray(data, dtype=np.float32)
                    if mono.size == 0:
                        continue
                    if on_level is not None:
                        on_level(float(np.sqrt(np.mean(np.square(mono)))))
                    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
                    w.writeframes(pcm16.tobytes())
                    total_frames += mono.size
        except KeyboardInterrupt:
            pass  # Ctrl-C = "点完成"
    return total_frames / SAMPLE_RATE
