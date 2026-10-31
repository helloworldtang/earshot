"""B站字幕直拉:yt-dlp 提取元信息与字幕轨 → Segment 列表。

高信噪比路线:听录会混入环境噪音/被打断,字幕来自平台已生成文本,无此问题。
支持 .bcc/.json(B站)与 .srt/.vtt 格式;AI 字幕(ai-zh)通常需要登录态
(--cookies-from-browser)。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from earshot.transcribe import Segment

# 人工字幕优先于 AI 字幕;同组内先列者优先
PREFERRED_LANGS = ("zh-CN", "zh-Hans", "zh", "ai-zh")
PARSABLE_EXTS = ("bcc", "json", "srt", "vtt")

_SRT_TIME = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[.,](?P<ms>\d{3})")


class SubtitleError(RuntimeError):
    """字幕获取失败(无字幕/网络/解析)。"""


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """视频元信息(写进笔记脚注)。

    Attributes:
        title: 视频标题。
        uploader: UP主/作者。
        duration_s: 视频时长(秒,缺失为 0)。
        url: 原始链接(回溯用)。
        subtitle_lang: 实际采用的字幕语言(如 zh-CN / ai-zh)。
    """

    title: str
    uploader: str
    duration_s: float
    url: str
    subtitle_lang: str


@dataclass(frozen=True, slots=True)
class _Track:
    lang: str
    url: str
    ext: str


def choose_subtitle(tracks: list[_Track]) -> _Track | None:
    """按语言优先级与格式可解析性选字幕轨(纯函数)。

    人工 zh 字幕 > ai-zh;同语言内 bcc/json 优先于 srt/vtt。
    """
    by_lang: dict[str, list[_Track]] = {}
    for t in tracks:
        if t.ext in PARSABLE_EXTS and t.url:
            by_lang.setdefault(t.lang, []).append(t)
    for lang in PREFERRED_LANGS:
        candidates = by_lang.get(lang)
        if candidates:
            for ext in ("bcc", "json", "srt", "vtt"):
                for t in candidates:
                    if t.ext == ext:
                        return t
            return candidates[0]
    return None


def parse_subtitle(content: str, ext: str) -> list[Segment]:
    """解析字幕内容为 Segment 列表(纯函数)。

    Args:
        content: 字幕文件文本。
        ext: bcc/json(B站 JSON 体)或 srt/vtt。

    Returns:
        按时间排序、非空文本的 Segment 列表。

    Raises:
        SubtitleError: 格式无法解析。
    """
    if ext in ("bcc", "json"):
        try:
            body = json.loads(content).get("body") or []
        except ValueError as e:
            msg = f"bcc/json 字幕解析失败: {e}"
            raise SubtitleError(msg) from e
        bcc_segments = [
            Segment(start_s=float(item["from"]), end_s=float(item["to"]), text=str(item["content"]).strip())
            for item in body
            if item.get("content")
        ]
        return bcc_segments
    # srt / vtt
    segments: list[Segment] = []
    blocks = content.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        times = [ln for ln in lines if "-->" in ln]
        if not times:
            continue
        m_start = _SRT_TIME.search(times[0])
        m_end = _SRT_TIME.search(times[0], times[0].find("-->") + 3)
        text = " ".join(ln.strip() for ln in lines if "-->" not in ln and not ln.strip().isdigit())
        if m_start and m_end and text:
            segments.append(
                Segment(
                    start_s=_to_seconds(m_start),
                    end_s=_to_seconds(m_end),
                    text=text,
                )
            )
    if not segments:
        msg = f"srt/vtt 字幕解析失败(无有效块): ext={ext}"
        raise SubtitleError(msg)
    return segments


def _to_seconds(m: re.Match[str]) -> float:
    return int(m["h"]) * 3600 + int(m["m"]) * 60 + int(m["s"]) + int(m["ms"]) / 1000.0


def fetch_video_note(url: str, cookies_browser: str | None = None) -> tuple[VideoMeta, list[Segment]]:
    """拉取视频字幕,返回 (元信息, Segment 列表)。

    Args:
        url: 视频链接(B站等 yt-dlp 支持的平台)。
        cookies_browser: 浏览器名(chrome/safari/firefox/edge),拉 ai-zh AI 字幕
            通常需要登录态;None 则匿名尝试。

    Raises:
        SubtitleError: 提取失败或无可用字幕。
    """
    import yt_dlp  # noqa: PLC0415 延迟导入:不用 video 命令时不加载

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    try:
        info = yt_dlp.YoutubeDL(opts).extract_info(url, download=False)
    except Exception as e:  # yt_dlp 异常族庞大,统一收口
        msg = f"yt-dlp 提取失败: {e}"
        raise SubtitleError(msg) from e
    if not info:
        msg = "yt-dlp 未返回视频信息"
        raise SubtitleError(msg)

    tracks: list[_Track] = []
    for field in ("subtitles", "automatic_captions"):
        for lang, subs in (info.get(field) or {}).items():
            tracks += [_Track(lang=lang, url=str(s.get("url") or ""), ext=str(s.get("ext") or "")) for s in subs]
    track = choose_subtitle(tracks)
    if track is None:
        hint = "。试试 --cookies-from-browser chrome(AI 字幕通常需要登录态)" if not cookies_browser else ""
        msg = f"无可用中文字幕{hint}"
        raise SubtitleError(msg)
    try:
        resp = httpx.get(track.url, timeout=60.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except httpx.HTTPError as e:
        msg = f"字幕下载失败({track.lang}/{track.ext}): {e}"
        raise SubtitleError(msg) from e
    segments = parse_subtitle(resp.text, track.ext)
    meta = VideoMeta(
        title=str(info.get("title") or url),
        uploader=str(info.get("uploader") or info.get("channel") or "-"),
        duration_s=float(info.get("duration") or 0),
        url=url,
        subtitle_lang=track.lang,
    )
    return meta, segments
