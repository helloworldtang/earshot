"""实时旁听:录音线程边录边喂,后台线程 VAD 切段+解码,段闭合即出字幕。

录音不中断、转写不阻塞:两者经队列解耦,SenseVoice 单段解码远快于实时,
后台线程永远追得上。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from earshot.transcribe import Segment, TranscribeError, run_live


class LiveSession:
    """边录边转会话:push 喂 16k 音频块,finalize 收尾并返回全部 Segment。

    Args:
        transcriber: 已初始化的 LocalTranscriber(提供 VAD 工厂与解码器)。
        on_text: 每段文本就绪即回调(实时字幕;回调里别做慢活,会拖住解码线程)。
    """

    def __init__(self, transcriber: Any, on_text: Callable[[str], None] | None = None) -> None:
        self._transcriber = transcriber
        self._on_text = on_text
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._segments: list[Segment] = []
        self._err: BaseException | None = None
        self._thread = threading.Thread(target=self._consume, daemon=True)

    def start(self) -> None:
        """启动后台解码线程。"""
        self._thread.start()

    def push(self, samples: np.ndarray) -> None:
        """喂一块 16k mono float32 样本(录音线程调用,非阻塞)。"""
        self._q.put(samples)

    def finalize(self, timeout_s: float = 300.0) -> list[Segment]:
        """收尾:喂入结束哨兵,等尾部音频解码完,返回全部 Segment。

        Raises:
            TranscribeError: 后台线程曾出错。
        """
        self._q.put(None)
        self._thread.join(timeout_s)
        if self._err is not None:
            msg = f"实时转写线程出错: {self._err}"
            raise TranscribeError(msg) from self._err
        return self._segments

    def _consume(self) -> None:
        try:
            self._segments = run_live(
                vad=self._transcriber._new_vad(),
                chunks=self._iter_queue(),
                decode=self._transcriber._decode,
                on_segment=lambda seg: self._on_text(seg.text) if self._on_text else None,
            )
        except BaseException as e:  # 线程内不可抛,存起来由 finalize 抛
            self._err = e

    def _iter_queue(self) -> Iterator[np.ndarray]:
        """队列 -> 迭代器,收到哨兵即结束。"""
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item
