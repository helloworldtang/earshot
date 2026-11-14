"""交互向导单测:非 tty 的回退行为与裸 earshot 入口。"""

from pathlib import Path

import pytest

from earshot import cli
from earshot.config import Settings


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
