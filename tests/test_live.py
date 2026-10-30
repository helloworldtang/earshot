"""实时旁听核心单测:run_live 与 LiveSession(假 VAD/假解码,不依赖模型)。

FakeVad 语义:每次 accept_waveform(= 一个 VAD 窗,512 样本)即闭一段,
段样本就是该窗 —— 与 silero 的真实闭合时机不同,但足以驱动核心循环。
"""

from typing import Any

import numpy as np

from earshot.live import LiveSession
from earshot.transcribe import VAD_WINDOW, Segment, run_live


class _FakeSeg:
    def __init__(self, start: int, samples: np.ndarray) -> None:
        self.start = start
        self.samples = samples


class FakeVad:
    def __init__(self) -> None:
        self._pending: list[np.ndarray] = []
        self._fed = 0

    def accept_waveform(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float32)
        self._pending.append(arr)
        self._fed += len(arr)

    def empty(self) -> bool:
        return not self._pending

    @property
    def front(self) -> _FakeSeg:
        return _FakeSeg(self._fed - len(self._pending[0]), self._pending[0])

    def pop(self) -> None:
        self._pending.pop(0)

    def flush(self) -> None:  # 无未闭段概念
        return None

    def is_speech_detected(self) -> bool:
        return False


def _windows(n: int) -> np.ndarray:
    return np.zeros(n * VAD_WINDOW, dtype=np.float32)


def _decode(samples: np.ndarray) -> str:
    return f"w{len(samples) // VAD_WINDOW}"


def test_run_live_one_segment_per_window() -> None:
    got: list[Segment] = []
    segments = run_live(vad=FakeVad(), chunks=iter([_windows(3)]), decode=_decode, on_segment=got.append)
    assert [s.text for s in segments] == ["w1", "w1", "w1"]
    assert segments == got  # 回调与返回一致
    starts = [s.start_s for s in segments]
    assert starts == sorted(starts)  # 时间轴单调


def test_run_live_carry_tail_is_flushed() -> None:
    # 0.5 秒静音 = 15 个整窗 + 320 样本尾巴;尾巴走收尾 flush 也成一段
    tail = np.zeros(8000, dtype=np.float32)
    segments = run_live(vad=FakeVad(), chunks=iter([tail]), decode=_decode)
    assert len(segments) == 16
    assert segments[-1].text == "w0"  # 尾段不足一窗


def test_run_live_carry_spans_chunks() -> None:
    # 块与窗不对齐时,carry 跨块拼接不丢样本
    from collections.abc import Iterator

    a = np.zeros(700, dtype=np.float32)

    def gen() -> Iterator[np.ndarray]:
        yield a[:333]
        yield a[333:]

    segments = run_live(vad=FakeVad(), chunks=gen(), decode=_decode)
    # 700 = 1 整窗(跨两块拼成) + 188 尾;尾段由收尾 flush 闭出
    assert [s.text for s in segments] == ["w1", "w0"]


class _StubTranscriber:
    def _new_vad(self) -> Any:
        return FakeVad()

    def _decode(self, samples: np.ndarray) -> str:  # noqa: ARG002
        return "字幕"


def test_live_session_thread_and_finalize() -> None:
    seen: list[str] = []
    session = LiveSession(_StubTranscriber(), on_text=seen.append)
    session.start()
    session.push(_windows(32))  # 32 窗
    session.push(_windows(32))
    segments = session.finalize(timeout_s=10)
    assert len(segments) == 64
    assert seen == ["字幕"] * 64


def test_live_session_error_surfaced_at_finalize() -> None:
    class _Boom(_StubTranscriber):
        def _decode(self, samples: np.ndarray) -> str:
            raise RuntimeError("boom")

    import pytest

    from earshot.transcribe import TranscribeError

    session = LiveSession(_Boom(), on_text=None)
    session.start()
    session.push(_windows(1))
    with pytest.raises(TranscribeError):
        session.finalize(timeout_s=10)
