"""_do_deliver 结构回归:投递成功必须返回成功,不得误报"无凭据"。

背景:一次真实事故 —— else 分支误挂在 try 上(try-else 语义=无异常即执行),
投递成功后仍打印"✗ 无 exomind 凭据也无 exomind CLI"。本用例钉死该结构。
"""

from pathlib import Path
from typing import Any

import pytest

from earshot import cli
from earshot.config import Settings
from earshot.deliver import DeliveryResult


def _settings(api_key: str = "key-1") -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://api.example.com",
        exomind_api_key=api_key,
    )


def test_successful_api_delivery_returns_ok_not_false_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    md = tmp_path / "note.md"
    md.write_text("# t\n正文", encoding="utf-8")

    def fake_api(_md: Path, _title: str, _tags: list[str], _s: Settings) -> DeliveryResult:
        return DeliveryResult("delivered", "已入库:实体 3 个")

    monkeypatch.setattr(cli, "deliver_exomind_api", fake_api)
    rc = cli._do_deliver("t", "# t\n正文", md, ["旁听"], "exomind", None, _settings())
    out = capsys.readouterr().out
    assert rc == 0
    assert "✓ delivered" in out
    assert "无 exomind 凭据" not in out  # 事故现场:成功路径误报无凭据


def test_no_key_no_cli_reports_failure_with_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: None)
    rc = cli._do_deliver("t", "内容", md, [], "exomind", None, _settings(api_key=""))
    out = capsys.readouterr().out
    assert rc == 1
    assert "无 exomind 凭据也无 exomind CLI" in out
    assert "HOME=" in out  # 自诊断数据在场


def test_none_kind_skips(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli._do_deliver("t", "x", Path("/nonexistent.md"), [], "none", None, _settings())
    assert rc == 0
    assert "跳过投递" in capsys.readouterr().out


def test_webhook_missing_url_fails(tmp_path: Path) -> None:
    rc = cli._do_deliver("t", "x", tmp_path / "n.md", [], "webhook", None, _settings())
    assert rc == 1


def test_do_deliver_cli_fallback_when_no_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    md = tmp_path / "note.md"
    md.write_text("内容", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/exomind")
    calls: dict[str, Any] = {}

    def fake_cli(_md: Path, title: str, tags: list[str]) -> DeliveryResult:
        calls["args"] = (title, tags)
        return DeliveryResult("delivered", "via cli")

    monkeypatch.setattr(cli, "deliver_exomind", fake_cli)
    rc = cli._do_deliver("t", "内容", md, ["旁听"], "exomind", None, _settings(api_key=""))
    assert rc == 0
    assert calls["args"] == ("t", ["旁听"])
    assert "via cli" in capsys.readouterr().out
