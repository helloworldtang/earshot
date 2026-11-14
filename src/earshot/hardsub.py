"""硬字幕 OCR 抽帧:无音轨且字幕烧在画面里的视频 → Segment 列表。

listen(ASR)与 video(字幕轨)都走不通的场景:ASR 无输入、字幕轨 API 返回 0 条
是正确结果。本模块走纯视觉路线:ffmpeg 按采样率抽底部字幕条带帧 → RapidOCR
识别 → 相邻同文本去重合并 → 带时间轴的 Segment,复用既有 _run_pipeline 入库。

RapidOCR 延迟导入(可选依赖 [ocr] extra),不用 ocr 命令不加载。
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from earshot.extract import ffmpeg_exe
from earshot.transcribe import Segment

DEFAULT_FPS = 1.0  # 抽帧采样率:字幕停留通常 ≥1s,1fps 够用
BOTTOM_BAND_FRAC = 0.22  # 底部字幕条带高度占比(画面下方 22%)
DEFAULT_MIN_SCORE = 0.55  # 行级置信度过滤阈值:清晰硬字幕 >0.9,深底低对比落 0.5-0.7
MERGE_GAP_FRAMES = 2  # 相同文本允许的最大丢帧数,桥接零星漏识别
MERGE_SIMILARITY = 0.8  # 相邻帧文本相似度阈值:OCR 标点变体(‘/“、？/?)不破去重
MAX_FRAMES = 21600  # 抽帧硬上限(fps=1 时 6 小时),防长视频失控

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)")

OcrEngine = Callable[[Path], list[tuple[str, float]]]  # 帧图路径 → [(文本, 置信度), ...]
ProgressFn = Callable[[int, int], None]  # (已完成帧数, 总帧数)


class HardsubError(RuntimeError):
    """硬字幕 OCR 提取失败(引擎缺失/下载/ffmpeg/识别)。"""


@dataclass(frozen=True, slots=True)
class FrameOcr:
    """一帧的 OCR 结果。

    Attributes:
        t_s: 帧时间戳(秒,由帧号/采样率推出)。
        text: 该帧全部识别行拼接后、归一化的文本。
        score: 该帧最弱行得分(整帧质量取短板)。
    """

    t_s: float
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class DownloadedVideo:
    """URL 模式下载产物(不复用 VideoMeta:subtitle_lang 对 OCR 场景无意义)。

    Attributes:
        path: 已下载视频文件路径。
        title: 视频标题。
        uploader: UP主/作者。
        duration_s: 时长(秒,缺失为 0)。
    """

    path: Path
    title: str
    uploader: str
    duration_s: float


def is_url(s: str) -> bool:
    """输入是否为 http(s) 链接(纯函数)。"""
    return s.lower().startswith(("http://", "https://"))


def build_vf_chain(fps: float, region: str) -> str:
    """组 ffmpeg -vf 滤镜链(纯函数)。

    Args:
        fps: 抽帧采样率。
        region: ``bottom``(底部字幕条带,裁剪后 2x 放大——小字笔画高度是
            PP-OCR det 的关键输入条件)或 ``full``(整帧,不放大)。

    Returns:
        滤镜链字符串。

    Raises:
        HardsubError: region 非法。
    """
    if region == "bottom":
        top = 1.0 - BOTTOM_BAND_FRAC
        return f"fps={fps},crop=iw:ih*{BOTTOM_BAND_FRAC}:0:ih*{top},scale=iw*2:-1"
    if region == "full":
        return f"fps={fps}"
    msg = f"未知字幕区域: {region}(可选 bottom/full)"
    raise HardsubError(msg)


def normalize_ocr_text(s: str) -> str:
    """归一化 OCR 文本:去半角/全角空格与换行(纯函数)。

    中文 OCR 常在字间插空格,去掉后相邻帧文本一致性更高,利于去重合并。
    """
    return s.replace(" ", "").replace("　", "").replace("\n", "").replace("\r", "")


def parse_ffmpeg_duration(stderr: str) -> float:
    """从 ffmpeg stderr 解析 Duration 行(纯函数)。

    Returns:
        时长秒数;解析不到返回 0.0(时长只影响进度显示,不值得抛错)。
    """
    m = _DURATION_RE.search(stderr)
    if not m:
        return 0.0
    h, mn, s, cs = m.groups()
    return int(h) * 3600 + int(mn) * 60 + int(s) + int(cs) / 100.0


def _parse_ocr_result(raw: Any) -> list[tuple[str, float]]:
    """适配 RapidOCR 调用结果为 [(文本, 置信度), ...](纯函数,wire 契约)。

    兼容两种形态:rapidocr 3.x 的结果对象(有 txts/scores 属性;检测为空时
    txts 为 None)与老包 rapidocr-onnxruntime 1.x 的 ``[[box, text, score], ...]``
    列表;识别失败(无文本)返回空列表。
    """
    if raw is None:
        return []
    if hasattr(raw, "txts"):  # rapidocr 3.x:检测为空的帧 txts=None
        txts = raw.txts or ()
        scores = getattr(raw, "scores", None) or ()
        return [(str(t), float(s)) for t, s in zip(txts, scores, strict=False)]
    out: list[tuple[str, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            out.append((str(item[1]), float(item[2])))
    return out


def merge_frame_ocr(results: list[FrameOcr], fps: float, gap_frames: int = MERGE_GAP_FRAMES) -> list[Segment]:
    """相邻同文本帧去重合并为 Segment(纯函数,核心算法)。

    文本相同或相似(相似度 ≥ MERGE_SIMILARITY,吸收 OCR 标点变体 ‘/“、？/?)
    且间隔 ≤ (gap_frames+1)/fps → 延长当前段(至多 gap_frames 个采样点漏识别被
    自动桥接,保留首见文本);否则切段。``end_s = last_t + 1/fps``(帧时刻代表该
    瞬间画面,字幕至少持续到下一个采样点),并与下一段起点取 min 防微重叠。

    Args:
        results: 按时间升序、已滤除空文本的帧结果。
        fps: 抽帧采样率(帧间隔 = 1/fps)。
        gap_frames: 桥接容差(允许的最多漏识别帧数;漏 N 帧时间戳间隔为 (N+1)/fps)。

    Returns:
        合并后的 Segment 列表(按时间排序)。
    """
    if not results:
        return []
    max_gap = (gap_frames + 1) / fps
    segments: list[Segment] = []
    cur_text = results[0].text
    cur_first = results[0].t_s
    cur_last = results[0].t_s
    for r in results[1:]:
        similar = r.text == cur_text or difflib.SequenceMatcher(None, cur_text, r.text).ratio() >= MERGE_SIMILARITY
        if similar and r.t_s - cur_last <= max_gap + 1e-9:
            cur_last = r.t_s
            continue
        segments.append(Segment(start_s=cur_first, end_s=cur_last + 1.0 / fps, text=cur_text))
        cur_text, cur_first, cur_last = r.text, r.t_s, r.t_s
    segments.append(Segment(start_s=cur_first, end_s=cur_last + 1.0 / fps, text=cur_text))
    for i in range(len(segments) - 1):  # 防与下一段微重叠
        if segments[i].end_s > segments[i + 1].start_s:
            patched = segments[i]
            segments[i] = Segment(patched.start_s, segments[i + 1].start_s, patched.text)
    return segments


def probe_duration(video: Path) -> float:
    """探测视频时长(秒):ffmpeg -i 读 stderr 的 Duration 行。

    项目无 ffprobe(imageio-ffmpeg 只带 ffmpeg),此为最省事的时长探测。
    returncode 非 0 属预期(没有输出参数)。

    Raises:
        HardsubError: ffmpeg 缺失或执行失败。
    """
    exe = ffmpeg_exe()
    if exe is None:
        msg = "未找到 ffmpeg(系统 PATH 与 imageio-ffmpeg 内置均无;重装 earshot 依赖)"
        raise HardsubError(msg)
    cmd = [exe, "-hide_banner", "-i", str(video)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # noqa: S603
    return parse_ffmpeg_duration(proc.stderr or "")


def extract_frames(video: Path, out_dir: Path, fps: float, region: str) -> list[tuple[float, Path]]:
    """整段抽帧为 jpg 序列,返回 [(时间戳, 帧路径), ...]。

    一次 ffmpeg 跑完,时间戳由帧号精确推出(避免分批 -ss 的关键帧定位漂移,
    漂移会破坏 group_paragraphs 的 2.5s 间隔分段语义)。上限 MAX_FRAMES 截断。

    Raises:
        HardsubError: ffmpeg 缺失或执行失败。
    """
    exe = ffmpeg_exe()
    if exe is None:
        msg = "未找到 ffmpeg(系统 PATH 与 imageio-ffmpeg 内置均无;重装 earshot 依赖)"
        raise HardsubError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",  # 不吞终端输入
        "-y",
        "-i",
        str(video),
        "-vf",
        build_vf_chain(fps, region),
        "-frames:v",
        str(MAX_FRAMES),
        "-q:v",
        "3",
        "-start_number",
        "0",
        str(out_dir / "f%06d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # noqa: S603
    if proc.returncode != 0:
        msg = f"抽帧失败({proc.returncode}): {(proc.stderr or '').strip()[:500]}"
        raise HardsubError(msg)
    frames = sorted(out_dir.glob("f*.jpg"))
    return [(i / fps, p) for i, p in enumerate(frames)]


def ocr_frames(
    frames: list[tuple[float, Path]],
    engine: OcrEngine,
    min_score: float = DEFAULT_MIN_SCORE,
    on_progress: ProgressFn | None = None,
    on_frame: Callable[[FrameOcr], None] | None = None,
) -> list[FrameOcr]:
    """逐帧 OCR,过滤低分行,每帧处理完立即删(磁盘峰值≈单帧)。

    Args:
        frames: extract_frames 产物。
        engine: OCR 引擎(注入,测试传闭包)。
        min_score: 行级置信度阈值,低于则整行丢弃(噪声文本进笔记比漏一句更伤阅读)。
        on_progress: 进度回调(每帧一次)。
        on_frame: 每个非空识别结果的回调(实时字幕上屏用,listen 同款体验)。

    Returns:
        非空文本的 FrameOcr 列表(按输入顺序)。
    """
    total = len(frames)
    out: list[FrameOcr] = []
    for done, (t_s, path) in enumerate(frames):
        lines = [(t, s) for t, s in engine(path) if s >= min_score and t.strip()]
        path.unlink(missing_ok=True)  # 逐帧删除:兑现磁盘紧张约束
        text = normalize_ocr_text("".join(t for t, _ in lines))
        if text:
            frame_ocr = FrameOcr(t_s=t_s, text=text, score=min((s for _, s in lines), default=0.0))
            out.append(frame_ocr)
            if on_frame:
                on_frame(frame_ocr)
        if on_progress:
            on_progress(done + 1, total)
    return out


def make_rapidocr_engine() -> OcrEngine:
    """构造 RapidOCR 引擎闭包(模型只加载一次,全视频复用)。

    Returns:
        OcrEngine:帧路径 → [(文本, 置信度), ...]。

    Raises:
        HardsubError: [ocr] extra 未安装。
    """
    try:
        from rapidocr import RapidOCR  # noqa: PLC0415 延迟导入:不用 ocr 命令不加载
    except ImportError as e:
        msg = 'OCR 引擎未安装。安装:uv tool install "earshot-cli[ocr]"(或 pip install "earshot-cli[ocr]")'
        raise HardsubError(msg) from e
    import logging

    # INFO/WARNING 会打断 \r 进度行;须在构造之后设——RapidOCR() 初始化时会重配自己的 logger
    ocr = RapidOCR()
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)

    def run(frame: Path) -> list[tuple[str, float]]:
        return _parse_ocr_result(ocr(str(frame)))

    return run


def download_video(url: str, dst_dir: Path, cookies_browser: str | None = None) -> DownloadedVideo:
    """yt-dlp 下载视频到 dst_dir(合并为 mp4),返回路径与元信息。

    Raises:
        HardsubError: 下载失败。
    """
    import yt_dlp  # noqa: PLC0415 延迟导入:不用 ocr 命令不加载

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # 无音轨视频(本场景触发源)自动落到单流分支,无需合并
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dst_dir / f"ocr-{ts}.%(ext)s"),
    }
    if exe := ffmpeg_exe():
        opts["ffmpeg_location"] = exe  # 系统无 PATH ffmpeg 时 yt-dlp 也能完成合并
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    try:
        info = yt_dlp.YoutubeDL(opts).extract_info(url, download=True)
    except Exception as e:  # yt_dlp 异常族庞大,统一收口
        for part in dst_dir.glob(f"ocr-{ts}.*"):  # 失败也清掉 .part 半成品
            part.unlink(missing_ok=True)
        msg = f"视频下载失败: {e}"
        raise HardsubError(msg) from e
    if not info:
        msg = "yt-dlp 未返回视频信息"
        raise HardsubError(msg)
    files = sorted(dst_dir.glob(f"ocr-{ts}.*"))
    videos = [f for f in files if f.suffix.lower() != ".part"]
    if not videos:
        msg = f"下载完成但未找到产物: {dst_dir}/ocr-{ts}.*"
        raise HardsubError(msg)
    return DownloadedVideo(
        path=videos[0],
        title=str(info.get("title") or url),
        uploader=str(info.get("uploader") or info.get("channel") or "-"),
        duration_s=float(info.get("duration") or 0),
    )


def extract_hardsub(
    video: Path,
    tmp_root: Path,
    fps: float = DEFAULT_FPS,
    region: str = "bottom",
    min_score: float = DEFAULT_MIN_SCORE,
    engine: OcrEngine | None = None,
    on_progress: ProgressFn | None = None,
    on_frame: Callable[[FrameOcr], None] | None = None,
) -> tuple[float, list[Segment]]:
    """顶层编排:抽帧 → OCR → 合并,返回 (视频时长, Segment 列表)。

    帧目录在 finally 中删除;不删视频本体(URL/本地的删除策略差异留给 cmd 层)。
    on_frame 透传给 ocr_frames(实时字幕上屏)。

    Raises:
        HardsubError: 引擎/ffmpeg 任一环节失败。
    """
    if engine is None:
        engine = make_rapidocr_engine()
    duration_s = probe_duration(video)
    est = int(duration_s * fps) if duration_s else 0
    if est:
        print(f"→ 视频时长 {duration_s:.0f}s,采样 {fps}fps ≈ {min(est, MAX_FRAMES)} 帧")
    frames_dir = tmp_root / f"ocr-frames-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        frames = extract_frames(video, frames_dir, fps, region)
        if not frames:
            msg = f"未抽出任何帧(视频损坏或时长为 0): {video}"
            raise HardsubError(msg)
        ocr_results = ocr_frames(frames, engine, min_score, on_progress, on_frame)
        return duration_s, merge_frame_ocr(ocr_results, fps)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
