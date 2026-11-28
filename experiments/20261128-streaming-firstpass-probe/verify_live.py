"""真引擎端到端:LocalTranscriber + run_live(partial) 走 listen 同路径,验证粗稿/终稿双通道。

用法:uv run python experiments/20261128-streaming-firstpass-probe/verify_live.py <wav16k>
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from earshot.config import load_settings  # noqa: E402
from earshot.transcribe import LocalTranscriber, run_live  # noqa: E402

CHUNK = 1600  # 100ms


def main() -> int:
    wav = Path(sys.argv[1])
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    model_root = load_settings().model_root
    t = LocalTranscriber(model_root)
    feed = t._new_partial_feed()
    assert feed is not None, "流式粗稿模型缺失"

    partials: list[str] = []
    finals: list[str] = []
    last = ""
    n_reset = 0

    def on_partial(text: str) -> None:
        nonlocal last
        partials.append(text)
        last = text

    orig_reset = feed.reset

    def counting_reset() -> None:
        nonlocal n_reset, last
        n_reset += 1
        orig_reset()
        last = ""

    feed.reset = counting_reset  # type: ignore[method-assign] 计数包装,验证 reset 次数

    segments = run_live(
        vad=t._new_vad(),
        chunks=(samples[i : i + CHUNK] for i in range(0, len(samples), CHUNK)),
        decode=t._decode,
        on_segment=lambda seg: finals.append(seg.text),
        partial_feed=feed,
        on_partial=on_partial,
        max_speech_run_s=10.0,
    )

    print(f"粗稿事件 {len(partials)} 个,粗稿流重置 {n_reset} 次(=段数 {len(segments)})")
    print(f"末次粗稿:{partials[-1][:60]}…")
    print("终稿(逐段):")
    for s in segments:
        print(f"  [{s.start_s:>5.1f}-{s.end_s:>5.1f}] {s.text}")

    assert len(partials) > 0, "粗稿通道没触发"
    assert n_reset == len(segments), f"reset 次数 {n_reset} != 段数 {len(segments)}"
    # 回归对拍基线 = 同预算、无粗稿件的 run_live:粗稿挂载不得改变终稿切分与文本
    baseline = run_live(
        vad=t._new_vad(),
        chunks=(samples[i : i + CHUNK] for i in range(0, len(samples), CHUNK)),
        decode=t._decode,
        max_speech_run_s=10.0,
    )
    assert [s.text for s in baseline] == finals, "带粗稿件与不带粗稿件的终稿不一致:挂载点改动了终稿路径"
    print("\n✓ 粗稿通道触发、逐段重置、终稿与无粗稿基线逐字一致(两遍识别全链通)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
