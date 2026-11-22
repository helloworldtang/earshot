"""交互向导单测:非 tty 的回退行为与裸 earshot 入口。"""

from datetime import datetime
from pathlib import Path

import pytest

from earshot import cli
from earshot.config import Settings
from earshot.transcribe import Segment


def _settings() -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://api.example.com",
        exomind_api_key="",
    )


@pytest.fixture()
def non_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)


def test_bare_earshot_non_tty_prints_help_and_fails(non_tty, capsys) -> None:  # type: ignore[no-untyped-def]
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "listen" in out  # 打印了用法


def test_video_without_url_non_tty_errors(non_tty, capsys) -> None:  # type: ignore[no-untyped-def]
    args = cli.build_parser().parse_args(["video"])
    rc = cli.cmd_video(args, _settings())
    assert rc == 1
    assert "缺少视频链接" in capsys.readouterr().out


def test_audio_without_file_non_tty_errors(non_tty, capsys) -> None:  # type: ignore[no-untyped-def]
    args = cli.build_parser().parse_args(["audio"])
    rc = cli.cmd_audio(args, _settings())
    assert rc == 1
    assert "缺少文件路径" in capsys.readouterr().out


def test_ocr_without_target_non_tty_errors(non_tty, capsys) -> None:  # type: ignore[no-untyped-def]
    args = cli.build_parser().parse_args(["ocr"])
    rc = cli.cmd_ocr(args, _settings())
    assert rc == 1
    assert "缺少视频路径或链接" in capsys.readouterr().out


def test_ocr_local_file_missing_fails(capsys) -> None:  # type: ignore[no-untyped-def]
    args = cli.build_parser().parse_args(["ocr", "/nonexistent/video.mp4"])
    rc = cli.cmd_ocr(args, _settings())
    assert rc == 1
    assert "文件不存在" in capsys.readouterr().out


def test_positionals_still_accept_args() -> None:
    parser = cli.build_parser()
    a = parser.parse_args(["video", "https://x"])
    assert a.url == "https://x" and a.cookies_from_browser is None
    b = parser.parse_args(["audio", "/tmp/a.mp4"])
    assert b.file == Path("/tmp/a.mp4")
    c = parser.parse_args(["ocr", "https://x", "--fps", "0.5", "--region", "full"])
    assert c.path_or_url == "https://x" and c.fps == 0.5 and c.region == "full" and c.min_score == 0.55


def test_clipboard_text_returns_str_without_crash() -> None:
    out = cli._clipboard_text()
    assert isinstance(out, str)


def test_short_url_strips_tracking_and_truncates() -> None:
    long_q = "https://www.bilibili.com/video/BV16qpF64E6V/?trackid=web_pegasus_0.router-2479516&spm_id_from=333.1007"
    assert cli._short_url(long_q) == "https://www.bilibili.com/video/BV16qpF64E6V/"
    assert cli._short_url("https://" + "a" * 100).endswith("…")
    assert len(cli._short_url("https://" + "a" * 100)) == 61


def test_video_no_subtitle_tracks_hints_ocr(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """字幕轨 0 条(硬字幕)时不能空手而归:给出可执行的 ocr 改走指引。"""
    from earshot.subtitle import VideoMeta

    meta = VideoMeta(title="硬字幕视频", uploader="u", duration_s=1.0, url="https://b23.tv/x", subtitle_lang="")
    monkeypatch.setattr(cli, "fetch_video_note", lambda _url, _cookies: (meta, []))
    args = cli.build_parser().parse_args(["video", "https://b23.tv/x", "-y"])
    rc = cli.cmd_video(args, _settings())
    out = capsys.readouterr().out
    assert rc == cli.EXIT_NO_CONTENT
    assert "earshot ocr https://b23.tv/x" in out


def test_run_pipeline_default_tag_follows_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """默认 tag 按摄入路线走(video→视频字幕),不再清一色「旁听」。"""
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_do_deliver", lambda *a: captured.update(tags=a[3]) or 0)
    args = cli.build_parser().parse_args(["video", "https://x", "-y", "--out", str(tmp_path)])
    meta = cli.NoteMeta(started_at=datetime.now(), duration_s=1.0, device="-", source="video", note="")
    rc = cli._run_pipeline([Segment(0.0, 1.0, "内容")], meta, args, _settings())
    assert rc == 0
    assert captured["tags"] == ["视频字幕"]


def _settings_home(home: Path) -> Settings:
    return Settings(
        home=home,
        model_root=home / "models",
        exomind_base_url="https://api.example.com",
        exomind_api_key="",
    )


def test_resolve_cookies_asks_once_then_remembers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """交互问一次后落盘记忆:第二次起不再问,静默沿用上次选择。"""
    settings = _settings_home(tmp_path)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: "safari")
    assert cli._resolve_cookies(None, settings) == "safari"
    prompts: list[object] = []
    monkeypatch.setattr(cli, "_prompt", lambda *a, **k: prompts.append(a) or "chrome")
    assert cli._resolve_cookies(None, settings) == "safari"  # 记忆生效
    assert not prompts


def test_resolve_cookies_explicit_none_overrides_and_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """显式 --cookies-from-browser none 走匿名并覆盖记忆;之后无旗标也沿用匿名。"""
    from earshot.config import load_state

    settings = _settings_home(tmp_path)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli._resolve_cookies("none", settings) is None
    assert load_state(settings.home)["cookies_browser"] == ""
    assert cli._resolve_cookies(None, settings) is None  # 匿名也被记住,且不再追问
