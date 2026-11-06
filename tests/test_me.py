"""cmd_me 测试:凭据状态展示三分支(有效/无效/未配置),探活注入不联网。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from earshot.cli import cmd_me
from earshot.config import Settings


def _settings(api_key: str = "") -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://exomind.example",
        exomind_api_key=api_key,
    )


def test_me_valid_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("earshot.cli.verify_key", lambda base, key: True)
    code = cmd_me(argparse.Namespace(), _settings("sk_live_valid_key"))
    out = capsys.readouterr().out
    assert code == 0
    assert "https://exomind.example" in out
    assert "sk_live_…_key" in out, "凭据掩码展示"
    assert "凭据状态:有效(长期 API Key,无到期时间)" in out, "长期 key 无 TTL,口径对齐 wxpub sk_ 分支"


def test_me_invalid_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("earshot.cli.verify_key", lambda base, key: False)
    code = cmd_me(argparse.Namespace(), _settings("sk_live_invalid_"))
    out = capsys.readouterr().out
    assert code == 1
    assert "earshot login --force" in out, "无效要指路重登"


def test_me_no_key(capsys: pytest.CaptureFixture[str]) -> None:
    code = cmd_me(argparse.Namespace(), _settings(""))
    out = capsys.readouterr().out
    assert code == 1
    assert "未配置" in out and "earshot login" in out
