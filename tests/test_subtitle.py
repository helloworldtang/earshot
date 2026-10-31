"""subtitle 模块单测:bcc/srt 解析 + 字幕轨选择(纯函数,无网络)。"""

from earshot.subtitle import _Track, choose_subtitle, parse_subtitle
from earshot.transcribe import Segment

BCC = """{"body": [
    {"from": 0.0, "to": 2.5, "content": "知识飞轮的核心是复利回路"},
    {"from": 2.6, "to": 5.0, "content": "每次存入都会抽取实体和关系"},
    {"from": 5.1, "to": 6.0, "content": ""}
]}"""

SRT = """1
00:00:01,000 --> 00:00:03,500
第一句

2
00:00:04,000 --> 00:00:06,000
第二句
"""


def test_parse_bcc_drops_empty() -> None:
    segments = parse_subtitle(BCC, "bcc")
    assert [s.text for s in segments] == ["知识飞轮的核心是复利回路", "每次存入都会抽取实体和关系"]
    assert segments[0].start_s == 0.0
    assert segments[1].end_s == 5.0


def test_parse_srt_timestamps_and_text() -> None:
    segments = parse_subtitle(SRT, "srt")
    assert [s.text for s in segments] == ["第一句", "第二句"]
    assert segments[0].start_s == 1.0 and segments[0].end_s == 3.5
    assert segments[1].start_s == 4.0


def test_parse_srt_handles_vtt_dots() -> None:
    vtt = SRT.replace(",", ".")
    segments = parse_subtitle(vtt, "vtt")
    assert len(segments) == 2


def test_choose_prefers_manual_over_ai() -> None:
    tracks = [
        _Track(lang="ai-zh", url="https://x/ai.bcc", ext="bcc"),
        _Track(lang="zh-CN", url="https://x/manual.srt", ext="srt"),
    ]
    assert choose_subtitle(tracks) == _Track("zh-CN", "https://x/manual.srt", "srt")


def test_choose_prefers_bcc_ext_within_lang() -> None:
    tracks = [
        _Track(lang="ai-zh", url="https://x/a.srt", ext="srt"),
        _Track(lang="ai-zh", url="https://x/b.bcc", ext="bcc"),
    ]
    assert choose_subtitle(tracks) == _Track("ai-zh", "https://x/b.bcc", "bcc")


def test_choose_returns_none_when_no_parsable() -> None:
    assert choose_subtitle([_Track(lang="en", url="https://x/v.vtt", ext="vtt")]) is None
    assert choose_subtitle([]) is None


def test_segment_dataclass_smoke() -> None:
    seg = Segment(start_s=0, end_s=1, text="t")
    assert (seg.start_s, seg.end_s, seg.text) == (0, 1, "t")
