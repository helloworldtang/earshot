"""本地转写引擎:silero VAD 切段 + SenseVoice 离线解码。

sherpa-onnx 延迟导入 —— compose 等纯逻辑模块与单测不依赖模型栈。
"""

from __future__ import annotations

import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SAMPLE_RATE = 16000
# SenseVoice 支持的语种;auto=逐段自动判语种,实时旁听的超短片段上会把中文误判成
# 日/韩(zh↔ja↔ko 声学最易混,SenseVoice 已知问题),故默认钉死 zh 跳过语种分类。
SENSE_VOICE_LANGS = ("auto", "zh", "en", "ja", "ko", "yue")
DEFAULT_LANG = "zh"
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

    def __init__(self, model_root: Path, language: str = DEFAULT_LANG) -> None:
        """加载 ASR/VAD 模型。

        Args:
            model_root: 模型根目录,须含 ``asr/model.int8.onnx``、``asr/tokens.txt``、
                ``vad/silero_vad.onnx``(scripts/download_models.sh 产物)。
            language: 转写语种,取值见 ``SENSE_VOICE_LANGS``。默认钉死 zh(跳过逐段
                语种分类,根治短段日/韩误判);纯外语音频才用 auto。

        Raises:
            TranscribeError: 语种非法、模型缺失或初始化失败。
        """
        if language not in SENSE_VOICE_LANGS:
            msg = f"语种非法: {language}(可选 {'/'.join(SENSE_VOICE_LANGS)})"
            raise TranscribeError(msg)
        asr_model = model_root / "asr" / "model.int8.onnx"
        tokens = model_root / "asr" / "tokens.txt"
        vad_model = model_root / "vad" / "silero_vad.onnx"
        for p in (asr_model, tokens, vad_model):
            if not p.is_file():
                msg = f"模型缺失: {p}(先跑 scripts/download_models.sh,或设 EARSHOT_MODEL_ROOT 指向已有模型)"
                raise TranscribeError(msg)
        self._model_root = model_root  # 粗稿供给器(_new_partial_feed)按需定位流式模型

        import sherpa_onnx  # noqa: PLC0415 延迟导入:纯逻辑模块/单测不拉模型栈

        try:
            self._recognizer: Any = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(asr_model),
                tokens=str(tokens),
                language=language,
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

    def _new_partial_feed(self) -> PartialFeed | None:
        """新建实时粗稿供给器(两遍识别首遍)。

        流式模型缺失时返回 None:实时字幕降级为仅终稿(闭段才出),批量路线不受影响。
        """
        stream_dir = self._model_root / "asr_stream"
        required = (
            "tokens.txt",
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
        )
        if any(not (stream_dir / p).is_file() for p in required):
            return None
        return PartialFeed(self._model_root)


class PartialFeed:
    """流式粗稿供给器:zipformer 双语逐窗增量解码,段闭合后 reset() 重开流。

    两遍识别的首遍:粗稿只求快(说到的词百 ms 级上屏,同音字糙可接受),
    准确性由 SenseVoice 整段重解码的终稿负责。粗稿无标点属预期。
    """

    def __init__(self, model_root: Path) -> None:
        import sherpa_onnx  # noqa: PLC0415 延迟导入:纯逻辑模块/单测不拉模型栈

        d = model_root / "asr_stream"
        self._recognizer: Any = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=2,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=False,  # 闭段由 silero VAD 驱动,粗稿流只管增量解码
            decoding_method="greedy_search",
        )
        self._stream: Any = self._recognizer.create_stream()

    def __call__(self, window: np.ndarray) -> str:
        """喂一窗(512 样本),返回当前粗稿(可能为空串)。"""
        self._stream.accept_waveform(SAMPLE_RATE, window)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        return str(self._recognizer.get_result(self._stream)).strip()

    def reset(self) -> None:
        """段闭合后重开流:粗稿从下一句重新开始,不背上一句的尾巴。"""
        self._stream = self._recognizer.create_stream()


class PartialFeeder(Protocol):
    """run_live 依赖的粗稿接口(PartialFeed 与单测假件的结构协议)。"""

    def __call__(self, window: np.ndarray) -> str:
        """喂一窗音频,返回当前粗稿文本。"""
        ...

    def reset(self) -> None:
        """段闭合后重开粗稿流。"""
        ...


def run_live(
    vad: Any,
    chunks: Iterator[np.ndarray],
    decode: Callable[[np.ndarray], str],
    on_segment: Callable[[Segment], None] | None = None,
    on_progress: Any = None,
    total_samples: int | None = None,
    max_speech_run_s: float = MAX_SPEECH_RUN_S,
    partial_feed: PartialFeeder | None = None,
    on_partial: Callable[[str], None] | None = None,
) -> list[Segment]:
    """VAD 切段 + 逐段解码核心:批量转写与实时旁听共用。

    Args:
        vad: sherpa VoiceActivityDetector(状态机,每次调用须新建)。
        chunks: 16k mono float32 音频块迭代器,逐块喂入。
        decode: 单段样本 -> 文本(终稿)。
        on_segment: 每段解码完成即回调(实时字幕终稿用)。
        on_progress: 进度回调(需 total_samples;实时模式无总量则不回调)。
        total_samples: 音频总样本数(进度计算用,实时模式为 None)。
        max_speech_run_s: 连续无闭段语音超过该秒数强制闭段 —— 批量为防 VAD
            buffer 溢出(55s),实时为字幕滚动预算(10s):连续口播停顿全部低于
            min_silence 阈值时,VAD 永不自然闭段,字幕要等收尾 flush 才出。
        partial_feed: 流式粗稿供给器(实时两遍识别首遍;None=不产粗稿)。
        on_partial: 粗稿变化即回调(与 partial_feed 必须同给才启用)。

    Returns:
        按时间排序的 Segment 列表。
    """
    segments: list[Segment] = []
    fed = 0  # 已喂样本数(绝对时间锚点)
    last_pop = 0  # 上次弹出段时的已喂样本数
    next_progress = PARAGRAPH_FEED_EVERY_S * SAMPLE_RATE
    last_partial = ""  # 粗稿去重:内容没变就不重复上屏

    def drain() -> None:
        nonlocal last_pop, last_partial
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
            if partial_feed is not None and on_partial is not None:
                partial_feed.reset()  # 段闭合,粗稿流重开(终稿已上屏接管该行)
                last_partial = ""

    carry = np.empty(0, dtype=np.float32)
    for chunk in chunks:
        buf = (
            np.concatenate([carry, np.asarray(chunk, dtype=np.float32)])
            if carry.size
            else np.asarray(chunk, dtype=np.float32)
        )
        n_windows = len(buf) // VAD_WINDOW
        for w in range(n_windows):
            window = buf[w * VAD_WINDOW : (w + 1) * VAD_WINDOW]
            vad.accept_waveform(window)
            fed += VAD_WINDOW
            drain()
            if partial_feed is not None and on_partial is not None:
                text = partial_feed(window)
                if text and text != last_partial:
                    on_partial(text)
                    last_partial = text
            if bool(vad.is_speech_detected()) and (fed - last_pop) / SAMPLE_RATE > max_speech_run_s:
                vad.flush()  # 强制闭段:连续语音超预算(批量=防溢出,实时=字幕滚动)
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
