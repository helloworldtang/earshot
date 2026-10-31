"""转写段 → Markdown 笔记:分段、标题、元信息脚注、outbox 落盘。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from earshot.transcribe import Segment

PARAGRAPH_GAP_S = 2.5  # 段间隔超过此值即换段(自然段)
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


def group_paragraphs(segments: list[Segment], gap_s: float = PARAGRAPH_GAP_S) -> list[list[Segment]]:
    """按段间隔聚段为自然段。

    Args:
        segments: 时间排序的转写段。
        gap_s: 相邻段时间间隔超过该值则分段。

    Returns:
        段的列表(每个元素是一个自然段)。
    """
    paragraphs: list[list[Segment]] = []
    for seg in segments:
        if paragraphs and seg.start_s - paragraphs[-1][-1].end_s <= gap_s:
            paragraphs[-1].append(seg)
        else:
            paragraphs.append([seg])
    return paragraphs


SENTENCE_END_GAP_S = 0.8  # 段间隔超过此值视为一句话说完
_PUNCT_ENDINGS = "。！？；，、,.!?;:"


def add_punctuation(segments: list[Segment]) -> list[Segment]:
    """按时间轴给无标点转写恢复标点(B站 AI 字幕原文无标点;SenseVoice 已有,自动跳过)。

    规则:段尾已有标点不动;与下段时间间隔大→句号(问句语气词→问号),小→逗号;末段补句号。

    Args:
        segments: 时间排序的转写段。

    Returns:
        补齐标点的新 Segment 列表(原列表不改)。
    """
    out: list[Segment] = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        if text[-1] not in _PUNCT_ENDINGS:
            if i + 1 < len(segments):
                gap = segments[i + 1].start_s - seg.end_s
                if gap > SENTENCE_END_GAP_S:
                    text += "?" if text[-1] in ("吗", "呢", "吧") else "。"
                else:
                    text += ","
            else:
                text += "?" if text[-1] in ("吗", "呢", "吧") else "。"
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
    lines.extend(paragraphs)
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
