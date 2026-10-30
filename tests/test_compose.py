"""compose 模块单测:分段/标题/时长/渲染/outbox 路径。"""

from datetime import datetime
from pathlib import Path

from earshot.compose import (
    NoteMeta,
    derive_title,
    format_duration,
    group_paragraphs,
    outbox_path,
    render_markdown,
    slugify,
)
from earshot.transcribe import Segment


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(start_s=start, end_s=end, text=text)


def test_group_paragraphs_splits_on_gap() -> None:
    segs = [
        _seg(0.0, 2.0, "第一句。"),
        _seg(2.5, 5.0, "第二句,间隔 0.5s 同段。"),
        _seg(10.0, 12.0, "间隔 5s,另起一段。"),
    ]
    groups = group_paragraphs(segs)
    assert len(groups) == 2
    assert [s.text for s in groups[0]] == ["第一句。", "第二句,间隔 0.5s 同段。"]
    assert groups[1][0].text.startswith("间隔")


def test_group_paragraphs_empty() -> None:
    assert group_paragraphs([]) == []


def test_derive_title_override_wins() -> None:
    assert derive_title([_seg(0, 1, "随便")], "自定义标题") == "自定义标题"


def test_derive_title_truncates_long_first_segment() -> None:
    long_text = "这是一段非常长的开场白" * 5
    title = derive_title([_seg(0, 5, long_text)])
    assert len(title) == 41  # 40 字 + 省略号
    assert title.endswith("…")


def test_derive_title_no_content() -> None:
    assert derive_title([]) == "(无语音内容)"
    assert derive_title([_seg(0, 1, "   ")]) == "(无语音内容)"


def test_format_duration() -> None:
    assert format_duration(45) == "45秒"
    assert format_duration(754) == "12分34秒"
    assert format_duration(3723) == "1时02分03秒"


def test_slugify_keeps_cjk() -> None:
    assert slugify("知识飞轮:复利回路!") == "知识飞轮-复利回路"
    assert slugify("///") == "note"


def test_render_markdown_contains_footer() -> None:
    meta = NoteMeta(
        started_at=datetime(2026, 10, 30, 14, 0, 3),
        duration_s=754,
        device="BlackHole 2ch",
        source="listen",
        note="B站·第3期",
    )
    md = render_markdown(meta, ["第一段。", "第二段。"], title="测试标题")
    assert md.startswith("# 测试标题")
    assert "- 来源: 旁听录音" in md
    assert "- 开始: 2026-10-30 14:00:03" in md
    assert "- 时长: 12分34秒" in md
    assert "- 设备: BlackHole 2ch" in md
    assert "- 备注: B站·第3期" in md
    assert "第一段。" in md and "第二段。" in md


def test_render_markdown_omits_empty_note() -> None:
    meta = NoteMeta(started_at=datetime(2026, 1, 1), duration_s=10, device="-", source="audio", note="")
    md = render_markdown(meta, ["正文"], title="t")
    assert "备注" not in md


def test_outbox_path_pattern(tmp_path: Path) -> None:
    path = outbox_path(tmp_path, datetime(2026, 10, 30, 9, 5, 7), "知识飞轮入门")
    assert path == tmp_path / "20261030-090507-知识飞轮入门.md"
