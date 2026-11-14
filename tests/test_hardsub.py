"""hardsub 模块单测:纯函数 + 假引擎注入 + ffmpeg 真进程(无 ffmpeg 环境跳过)。"""

from pathlib import Path
from typing import Any

import pytest

from earshot.extract import ffmpeg_available, ffmpeg_exe
from earshot.hardsub import (
    DEFAULT_MIN_SCORE,
    FrameOcr,
    HardsubError,
    _parse_ocr_result,
    build_vf_chain,
    extract_frames,
    is_url,
    merge_frame_ocr,
    normalize_ocr_text,
    ocr_frames,
    parse_ffmpeg_duration,
    probe_duration,
)

# ---------- 纯函数 ----------


def test_is_url() -> None:
    assert is_url("https://www.bilibili.com/video/BV1yVn16XEz7")
    assert is_url("HTTP://example.com")
    assert not is_url("/tmp/a.mp4")
    assert not is_url("BV1yVn16XEz7")
    assert not is_url("./bv.mp4")


def test_build_vf_chain_bottom_and_full() -> None:
    chain = build_vf_chain(1.0, "bottom")
    assert chain == "fps=1.0,crop=iw:ih*0.22:0:ih*0.78,scale=iw*2:-1"
    assert build_vf_chain(0.5, "full") == "fps=0.5"  # full 不裁剪不放大
    with pytest.raises(HardsubError):
        build_vf_chain(1.0, "middle")


def test_normalize_ocr_text_strips_spaces_and_newlines() -> None:
    assert normalize_ocr_text("你 好 世 界") == "你好世界"
    assert normalize_ocr_text("全角　空格\n换行\r回车") == "全角空格换行回车"


def test_parse_ffmpeg_duration() -> None:
    stderr = "  Duration: 00:01:23.45, start: 0.000000, bitrate: 1234 kb/s"
    assert parse_ffmpeg_duration(stderr) == pytest.approx(83.45)
    assert parse_ffmpeg_duration("no duration here") == 0.0


def test_parse_ocr_result_v3_style_object() -> None:
    class FakeOutput:  # rapidocr 3.x:结果对象带 txts/scores
        txts = ("第一句", "第二句")
        scores = (0.91, 0.62)

    assert _parse_ocr_result(FakeOutput()) == [("第一句", 0.91), ("第二句", 0.62)]


def test_parse_ocr_result_legacy_list_style() -> None:
    # 老包 rapidocr-onnxruntime 1.x:[[box, text, score], ...]
    box = [[0, 0], [1, 0], [1, 1], [0, 1]]
    raw: Any = [[box, "旧格式", 0.88]]
    assert _parse_ocr_result(raw) == [("旧格式", 0.88)]
    assert _parse_ocr_result(None) == []


def test_parse_ocr_result_v3_empty_detection() -> None:
    # 真实踩坑(BV1yVn16XEz7):检测为空的帧 txts=None,不能落进老包迭代分支
    class FakeEmpty:
        txts = None
        scores = None

    assert _parse_ocr_result(FakeEmpty()) == []


# ---------- merge_frame_ocr(核心去重合并) ----------


def test_merge_frame_ocr_empty() -> None:
    assert merge_frame_ocr([], fps=1.0) == []


def test_merge_frame_ocr_merges_same_text() -> None:
    results = [FrameOcr(t_s=t, text="同一句", score=0.9) for t in (0.0, 1.0, 2.0)]
    segs = merge_frame_ocr(results, fps=1.0)
    assert len(segs) == 1
    assert segs[0].start_s == 0.0
    assert segs[0].end_s == pytest.approx(3.0)  # last_t(2.0) + 1/fps


def test_merge_frame_ocr_bridges_sparse_misses() -> None:
    # t=1、t=2 两帧漏识别:间隔 3s ≤ (gap_frames+1)/fps = 3s → 桥接
    results = [FrameOcr(t, "桥接句", 0.9) for t in (0.0, 3.0, 4.0)]
    segs = merge_frame_ocr(results, fps=1.0)
    assert len(segs) == 1
    assert segs[0].start_s == 0.0 and segs[0].end_s == pytest.approx(5.0)


def test_merge_frame_ocr_cuts_on_long_gap() -> None:
    # t=1..3 连续 3 帧漏识别:间隔 4s > (gap_frames+1)/fps = 3s → 切成两段
    results = [FrameOcr(t, "隔久了", 0.9) for t in (0.0, 4.0, 5.0)]
    segs = merge_frame_ocr(results, fps=1.0)
    assert len(segs) == 2
    assert segs[0].end_s == pytest.approx(1.0)  # 0.0 + 1/fps
    assert segs[1].start_s == 4.0


def test_merge_frame_ocr_cuts_on_text_change() -> None:
    results = [FrameOcr(0.0, "第一句", 0.9), FrameOcr(1.0, "第二句", 0.9)]
    segs = merge_frame_ocr(results, fps=1.0)
    assert [s.text for s in segs] == ["第一句", "第二句"]
    assert segs[0].end_s == pytest.approx(1.0)  # 不越过下一段 start


def test_merge_frame_ocr_merges_punctuation_variants() -> None:
    # 真实踩坑(BV1yVn16XEz7):OCR 把引号/问号识别成不同全半角变体,精确相等去重失效
    results = [
        FrameOcr(0.0, '心理学叫"互惠式自我暴露"，他敢进来', 0.9),
        FrameOcr(1.0, "心理学叫“互惠式自我暴露”，他敢进来", 0.9),
        FrameOcr(2.0, "这是什么吗?他敢进来", 0.9),  # 相似度低于阈值 → 切段
    ]
    segs = merge_frame_ocr(results, fps=1.0)
    assert len(segs) == 2
    assert segs[0].text == '心理学叫"互惠式自我暴露"，他敢进来'  # 保留首见文本


# ---------- ocr_frames(假引擎注入) ----------


def _make_fake_frames(tmp_path: Path, n: int = 3) -> list[tuple[float, Path]]:
    frames = []
    for i in range(n):
        p = tmp_path / f"f{i:06d}.jpg"
        p.write_bytes(b"fake-jpg")
        frames.append((float(i), p))
    return frames


def test_ocr_frames_filters_and_deletes(tmp_path: Path) -> None:
    calls: list[Path] = []

    def engine(path: Path) -> list[tuple[str, float]]:
        calls.append(path)
        return [("高分句", 0.9), ("低分噪", 0.2), ("", 0.99)]  # 低分与空文本都该被滤

    frames = _make_fake_frames(tmp_path)
    progress: list[tuple[int, int]] = []
    out = ocr_frames(frames, engine, min_score=DEFAULT_MIN_SCORE, on_progress=lambda d, t: progress.append((d, t)))
    assert len(out) == 3
    assert all(o.text == "高分句" for o in out)  # 空串整帧无文本则不出现——此处每帧仍有高分行
    assert all(not p.exists() for _, p in frames)  # 逐帧删除
    assert calls == [p for _, p in frames]
    assert progress == [(1, 3), (2, 3), (3, 3)]  # 3 帧全回调(粒度 32,少于粒度时逐帧)


def test_ocr_frames_empty_frame_skipped(tmp_path: Path) -> None:
    def engine(_path: Path) -> list[tuple[str, float]]:
        return []  # 整帧无识别结果

    frames = _make_fake_frames(tmp_path, n=2)
    assert ocr_frames(frames, engine) == []
    assert all(not p.exists() for _, p in frames)


def test_ocr_frames_on_frame_called_for_nonempty_only(tmp_path: Path) -> None:
    frames = _make_fake_frames(tmp_path, n=2)

    def engine(path: Path) -> list[tuple[str, float]]:
        return [("有内容", 0.9)] if path == frames[0][1] else []  # 第二帧无识别结果

    seen: list[str] = []
    out = ocr_frames(frames, engine, on_frame=lambda f: seen.append(f.text))
    assert seen == ["有内容"] and len(out) == 1


# ---------- ffmpeg 真进程 ----------


ffmpeg_skip = pytest.mark.skipif(not ffmpeg_available(), reason="本机无 ffmpeg")


def _make_testsrc(path: Path, seconds: int = 3) -> None:
    exe = ffmpeg_exe()
    assert exe is not None
    import subprocess

    subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@ffmpeg_skip
def test_probe_duration(tmp_path: Path) -> None:
    video = tmp_path / "src.mp4"
    _make_testsrc(video, seconds=3)
    assert probe_duration(video) == pytest.approx(3.0, abs=0.5)


@ffmpeg_skip
def test_extract_frames_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "src.mp4"
    _make_testsrc(video, seconds=3)
    frames = extract_frames(video, tmp_path / "frames", fps=1.0, region="bottom")
    assert [t for t, _ in frames] == pytest.approx([0.0, 1.0, 2.0])
    assert all(p.is_file() and p.name.startswith("f") for _, p in frames)
