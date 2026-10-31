"""转写段 → Markdown 笔记:分段、标题、元信息脚注、outbox 落盘。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from earshot.transcribe import Segment

PARAGRAPH_GAP_S = 2.5  # 段间隔超过此值即换段(自然段)
PARAGRAPH_MAX_CHARS = 400  # 单段上限,超出在句号处拆(口播字幕无长停顿时的兜底)
SENTENCE_MAX_CHARS = 60  # 句长预算:连续无句号超过该字数,强制断句
TITLE_MAX_CHARS = 40

SOURCE_LABELS = {"listen": "旁听录音", "audio": "音频文件", "video": "视频字幕"}


@dataclass(frozen=True, slots=True)
class NoteMeta:
    """笔记元信息(渲染进脚注)。

    Attributes:
        started_at: 录音开始时间。
        duration_s: 音频时长(秒)。
        device: 采集设备名(audio 文件模式为 ``-``)。
        source: ``listen`` 或 ``audio``。
        note: 来源备注(如"B站·xx视频"),可为空。
    """

    started_at: datetime
    duration_s: float
    device: str
    source: str
    note: str
    source_url: str = ""  # 原文链接(video 路线=视频 URL),投递时传服务端作 source_url 留痕


def group_paragraphs(
    segments: list[Segment],
    gap_s: float = PARAGRAPH_GAP_S,
    chapter_breaks: list[float] | None = None,
) -> list[list[Segment]]:
    """聚段为自然段:章节边界 > 时间间隔 > 长度兜底。

    Args:
        segments: 时间排序的转写段。
        gap_s: 相邻段时间间隔超过该值则分段。
        chapter_breaks: 视频章节起点(秒,B站章节标记),跨过即强制换段。

    Returns:
        段的列表(每个元素是一个自然段);单段不超过 PARAGRAPH_MAX_CHARS 字(句号处拆)。
    """
    chapters = sorted(chapter_breaks or [])
    paragraphs: list[list[Segment]] = []
    for seg in segments:
        make_new = not paragraphs or seg.start_s - paragraphs[-1][-1].end_s > gap_s
        if not make_new and chapters and paragraphs:
            prev_start = paragraphs[-1][0].start_s  # 现段起点跨过章节边界而本段首未跨 → 新段
            make_new = any(prev_start < cb <= seg.start_s for cb in chapters)
        if make_new:
            paragraphs.append([seg])
        else:
            paragraphs[-1].append(seg)

    result: list[list[Segment]] = []
    for para in paragraphs:
        if sum(len(s.text) for s in para) <= PARAGRAPH_MAX_CHARS:
            result.append(para)
            continue
        current: list[Segment] = []
        current_chars = 0
        for s in para:  # 超长段在句号处拆(口播字幕无长停顿时的兜底)
            current.append(s)
            current_chars += len(s.text)
            if current_chars >= PARAGRAPH_MAX_CHARS and s.text.rstrip().endswith(("。", "?", "!", "！", "？")):
                result.append(current)
                current = []
                current_chars = 0
        if current:
            result.append(current)
    return result


SENTENCE_END_GAP_S = 0.8  # 段间隔超过此值视为一句话说完
_PUNCT_ENDINGS = "。！？；，、,.!?;:"


def add_punctuation(segments: list[Segment]) -> list[Segment]:
    """按时间轴给无标点转写恢复标点(B站 AI 字幕原文无标点;SenseVoice 已有,自动跳过)。

    规则:段尾已有标点不动;与下段时间间隔大→句号(问句语气词→问号),小→逗号;末段补句号。
    句长预算:连续无句号超过 SENTENCE_MAX_CHARS 字时强制断句(口播字幕行间隔密集,
    固定间隔阈值会漏断,产生超长逗号串)。

    Args:
        segments: 时间排序的转写段。

    Returns:
        补齐标点的新 Segment 列表(原列表不改)。
    """
    out: list[Segment] = []
    chars_since_period = 0
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        chars_since_period += len(text)
        if text[-1] in _PUNCT_ENDINGS:
            if text[-1] in "。！？!?":
                chars_since_period = 0
        elif i + 1 < len(segments):
            gap = segments[i + 1].start_s - seg.end_s
            if gap > SENTENCE_END_GAP_S or chars_since_period >= SENTENCE_MAX_CHARS:
                text += "?" if text[-1] in ("吗", "呢", "吧") else "。"
                chars_since_period = 0
            else:
                text += ","
        else:
            text += "?" if text[-1] in ("吗", "呢", "吧") else "。"
            chars_since_period = 0
        out.append(Segment(start_s=seg.start_s, end_s=seg.end_s, text=text))
    return out


def derive_title(segments: list[Segment], override: str | None = None) -> str:
    """标题:``--title`` 覆盖优先,否则取首个非空段截断。

    Returns:
        标题文本;无任何内容时返回 ``"(无语音内容)"``。
    """
    if override:
        return override.strip()
    first = next((s.text.strip() for s in segments if s.text.strip()), "")
    first = first.rstrip(_PUNCT_ENDINGS)
    if not first:
        return "(无语音内容)"
    if len(first) > TITLE_MAX_CHARS:
        return first[:TITLE_MAX_CHARS].rstrip() + "…"
    return first


def format_duration(seconds: float) -> str:
    """时长人性化:``1时02分03秒`` / ``12分34秒`` / ``45秒``。"""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}时{m:02d}分{s:02d}秒"
    if m:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


def slugify(title: str) -> str:
    """标题 → 文件名安全片段(保留中文/字母/数字,其余折叠为 ``-``)。"""
    slug = re.sub(r"[^\w一-鿿]+", "-", title).strip("-")
    slug = slug[:30].strip("-")
    return slug or "note"


def render_markdown(meta: NoteMeta, paragraphs: list[str], title: str) -> str:
    """渲染完整 Markdown 笔记。

    Args:
        meta: 元信息。
        paragraphs: 已拼接好的自然段文本列表。
        title: 笔记标题(同时用于投递侧 ``--title``)。

    Returns:
        Markdown 字符串,含元信息脚注。
    """
    source_label = SOURCE_LABELS.get(meta.source, meta.source)
    lines = [f"# {title}", ""]
    for para in paragraphs:  # Markdown 相邻行会合并同段,段落间必须空行
        lines.append(para)
        lines.append("")
    lines += [
        "",
        "---",
        "",
        f"- 来源: {source_label}",
        f"- 开始: {meta.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 时长: {format_duration(meta.duration_s)}",
        f"- 设备: {meta.device}",
    ]
    if meta.note:
        lines.append(f"- 备注: {meta.note}")
    return "\n".join(lines).strip() + "\n"


def outbox_path(outbox: Path, started_at: datetime, title: str) -> Path:
    """outbox 文件路径:``YYYYMMDD-HHMMSS-<slug>.md``。"""
    return outbox / f"{started_at.strftime('%Y%m%d-%H%M%S')}-{slugify(title)}.md"
