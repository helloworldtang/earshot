"""本地转写引擎:silero VAD 切段 + SenseVoice 离线解码。

sherpa-onnx 延迟导入 —— compose 等纯逻辑模块与单测不依赖模型栈。
"""

from __future__ import annotations

import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
VAD_WINDOW = 512  # silero 要求 512 样本/窗(16k 下 32ms)
MAX_SPEECH_RUN_S = 55.0  # 连续无停顿语音强制切段,防 VAD 60s buffer 溢出
MIN_SILENCE_S = 0.45  # 短于此的静音不算说完(挡住句中停顿)
PARAGRAPH_FEED_EVERY_S = 5.0  # 进度回调最小间隔


class TranscribeError(RuntimeError):
    """本地转写失败(模型缺失 / 配置 / 解码错误)。"""


@dataclass(frozen=True, slots=True)
class Segment:
    """一段连续语音的转写结果。

    Attributes:
        start_s: 段起点(秒,相对音频开头)。
        end_s: 段终点(秒)。
        text: 转写文本。
    """

    start_s: float
    end_s: float
    text: str


class LocalTranscriber:
    """VAD 切段 + SenseVoice 批量识别,输入须为 16k mono wav。"""

    def __init__(self, model_root: Path) -> None:
        """加载 ASR/VAD 模型。

        Args:
            model_root: 模型根目录,须含 ``asr/model.int8.onnx``、``asr/tokens.txt``、
                ``vad/silero_vad.onnx``(scripts/download_models.sh 产物)。

        Raises:
            TranscribeError: 模型缺失或初始化失败。
        """
        asr_model = model_root / "asr" / "model.int8.onnx"
        tokens = model_root / "asr" / "tokens.txt"
        vad_model = model_root / "vad" / "silero_vad.onnx"
        for p in (asr_model, tokens, vad_model):
            if not p.is_file():
                msg = f"模型缺失: {p}(先跑 scripts/download_models.sh,或设 EARSHOT_MODEL_ROOT 指向已有模型)"
                raise TranscribeError(msg)

        import sherpa_onnx  # noqa: PLC0415 延迟导入:纯逻辑模块/单测不拉模型栈

        try:
            self._recognizer: Any = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(asr_model),
                tokens=str(tokens),
                use_itn=True,
            )
            self._vad_model_config: Any = sherpa_onnx.VadModelConfig()
            self._vad_model_config.silero_vad.model = str(vad_model)
            self._vad_model_config.silero_vad.threshold = 0.5
            self._vad_model_config.silero_vad.min_silence_duration = MIN_SILENCE_S
            self._vad_model_config.silero_vad.min_speech_duration = 0.15
            self._vad_model_config.silero_vad.window_size = VAD_WINDOW
            self._vad_model_config.sample_rate = SAMPLE_RATE
            if not self._vad_model_config.validate():
                msg = "VAD 配置校验失败"
                raise TranscribeError(msg)
        except TranscribeError:
            raise
        except Exception as e:  # sherpa 抛的均为未类型化异常
            msg = f"初始化本地转写失败: {e}"
            raise TranscribeError(msg) from e

    def transcribe(self, wav: Path, on_progress: Any = None) -> list[Segment]:
        """转写整个 wav 文件。

        Args:
            wav: 16k mono wav(earshot.extract.to_wav16k 产物)。
            on_progress: 可选回调 ``on_progress(fraction)``,fraction ∈ [0,1]。

        Returns:
            按时间排序的 Segment 列表。

        Raises:
            TranscribeError: wav 格式不符或解码失败。
        """
        samples = read_wav16k_mono(wav)
        return self.transcribe_samples(samples, on_progress)

    def transcribe_samples(self, samples: np.ndarray, on_progress: Any = None) -> list[Segment]:
        """转写内存样本(16k float32 mono)。"""
        return run_live(
            vad=self._new_vad(),
            chunks=iter([samples]),
            decode=self._decode,
            on_progress=on_progress,
            total_samples=len(samples),
        )

    def _new_vad(self) -> Any:
        """新建一个 VAD 状态机(有状态,每次转写须新建)。"""
        import sherpa_onnx  # noqa: PLC0415

        return sherpa_onnx.VoiceActivityDetector(self._vad_model_config, buffer_size_in_seconds=60)

    def _decode(self, samples: np.ndarray) -> str:
        """单段离线解码,返回去空白文本。"""
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self._recognizer.decode_stream(stream)
            return str(stream.result.text).strip()
        except Exception as e:
            msg = f"解码失败: {e}"
            raise TranscribeError(msg) from e


def run_live(
    vad: Any,
    chunks: Iterator[np.ndarray],
    decode: Callable[[np.ndarray], str],
    on_segment: Callable[[Segment], None] | None = None,
    on_progress: Any = None,
    total_samples: int | None = None,
) -> list[Segment]:
    """VAD 切段 + 逐段解码核心:批量转写与实时旁听共用。

    Args:
        vad: sherpa VoiceActivityDetector(状态机,每次调用须新建)。
        chunks: 16k mono float32 音频块迭代器,逐块喂入。
        decode: 单段样本 -> 文本。
        on_segment: 每段解码完成即回调(实时字幕用)。
        on_progress: 进度回调(需 total_samples;实时模式无总量则不回调)。
        total_samples: 音频总样本数(进度计算用,实时模式为 None)。

    Returns:
        按时间排序的 Segment 列表。
    """
    segments: list[Segment] = []
    fed = 0  # 已喂样本数(绝对时间锚点)
    last_pop = 0  # 上次弹出段时的已喂样本数
    next_progress = PARAGRAPH_FEED_EVERY_S * SAMPLE_RATE

    def drain() -> None:
        nonlocal last_pop
        while not vad.empty():
            seg = vad.front
            seg_samples = np.asarray(seg.samples, dtype=np.float32)
            text = decode(seg_samples)
            if text:
                start = seg.start / SAMPLE_RATE
                segment = Segment(start_s=start, end_s=start + len(seg_samples) / SAMPLE_RATE, text=text)
                segments.append(segment)
                if on_segment is not None:
                    on_segment(segment)
            vad.pop()
            last_pop = fed

    carry = np.empty(0, dtype=np.float32)
    for chunk in chunks:
        buf = (
            np.concatenate([carry, np.asarray(chunk, dtype=np.float32)])
            if carry.size
            else np.asarray(chunk, dtype=np.float32)
        )
        n_windows = len(buf) // VAD_WINDOW
        for w in range(n_windows):
            vad.accept_waveform(buf[w * VAD_WINDOW : (w + 1) * VAD_WINDOW])
            fed += VAD_WINDOW
            drain()
            if bool(vad.is_speech_detected()) and (fed - last_pop) / SAMPLE_RATE > MAX_SPEECH_RUN_S:
                vad.flush()  # 强制闭段:连续语音超限,防 buffer 溢出
                drain()
        carry = buf[n_windows * VAD_WINDOW :]
        if on_progress is not None and total_samples and fed >= next_progress:
            on_progress(fed / total_samples)
            next_progress = fed + PARAGRAPH_FEED_EVERY_S * SAMPLE_RATE
    if carry.size:
        vad.accept_waveform(carry)
        fed += carry.size
    vad.flush()
    drain()
    if on_progress is not None and total_samples:
        on_progress(1.0)
    return segments


def read_wav16k_mono(wav: Path) -> np.ndarray:
    """读 wav 为 16k mono float32 样本。

    Raises:
        TranscribeError: 非法 wav 或采样率/声道不符。
    """
    try:
        with wave.open(str(wav), "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
                msg = f"须为 16k mono wav(实际 {w.getframerate()}Hz/{w.getnchannels()}ch): {wav}"
                raise TranscribeError(msg)
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    except wave.Error as e:
        msg = f"不是合法 wav: {wav}({e})"
        raise TranscribeError(msg) from e
    return pcm.astype(np.float32) / 32768.0
