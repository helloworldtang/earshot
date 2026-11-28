"""探针:sherpa-onnx 流式 zipformer(双语 zh-en int8)在纯中文上的 partial 质量。

喂 100ms 块,记录每次 partial 变化(音频时刻/解码耗时/文本),末尾与
SenseVoice 终稿(SenseVoice Small INT8 + VAD 切段)对拍算 CER。

用法:uv run python experiments/20261128-streaming-firstpass-probe/probe.py <wav16k> <asr_stream_dir> <model_root>
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

CHUNK = 1600  # 100ms @16k


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, f"须 16k mono: {path}"
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def cer(ref: str, hyp: str) -> float:
    """字级编辑距离 / 参考长度(双方先去标点空白)。"""
    drop = set("，。、！？,.!?;；:：\"\"''\"' ")
    r = [c for c in ref if c not in drop]
    h = [c for c in hyp if c not in drop]
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def main() -> int:
    wav_path, stream_dir, model_root = (Path(p) for p in sys.argv[1:4])
    import sherpa_onnx  # noqa: PLC0415

    samples = read_wav(wav_path)
    total_s = len(samples) / 16000
    print(f"音频 {total_s:.1f}s,块 {CHUNK * 1000 // 16000}ms\n")

    t_load0 = time.monotonic()
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(stream_dir / "tokens.txt"),
        encoder=str(stream_dir / "encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(stream_dir / "decoder-epoch-99-avg-1.onnx"),
        joiner=str(stream_dir / "joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
    )
    print(f"流式模型加载 {time.monotonic() - t_load0:.1f}s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from earshot.transcribe import LocalTranscriber  # noqa: PLC0415

    t_load1 = time.monotonic()
    transcriber = LocalTranscriber(model_root)
    print(f"SenseVoice 加载 {time.monotonic() - t_load1:.1f}s\n")

    stream = recognizer.create_stream()
    events: list[tuple[float, float, str]] = []  # (音频时刻s, 累计解码耗时ms, partial)
    last = ""
    lag_sum = 0.0  # 逐块(喂+解码)耗时合计,衡量能否跟上实时
    t0 = time.monotonic()
    for i in range(0, len(samples), CHUNK):
        chunk = samples[i : i + CHUNK]
        t_feed = time.monotonic()
        stream.accept_waveform(16000, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        lag_sum += time.monotonic() - t_feed
        text = recognizer.get_result(stream)
        if text and text != last:
            events.append((round(i / 16000, 2), round((time.monotonic() - t0) * 1000), text))
            last = text

    print("== partial 时间线(每次变化一行) ==")
    for t_audio, t_wall, text in events:
        print(f"  音频{t_audio:>6.2f}s  墙钟{t_wall:>6}ms  {text}")
    final_partial = events[-1][2] if events else ""
    # >1 才追不上实时
    print(f"\npartial 变化次数 {len(events)}(刷新越频繁闪烁越多),流式侧总耗时 {lag_sum:.2f}s / 音频 {total_s:.1f}s")

    print("\n== SenseVoice 终稿(VAD 切段,生产同路径) ==")
    segs = transcriber.transcribe_samples(samples)
    finals = [s.text for s in segs]
    for s in segs:
        print(f"  [{s.start_s:>5.1f}-{s.end_s:>5.1f}] {s.text}")
    sv_text = "".join(finals)

    if sv_text:
        print(f"\n末次 partial vs 终稿 CER = {cer(sv_text, final_partial):.1%}  (粗稿: {final_partial})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
