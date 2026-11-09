"""凭据生命周期测试:update(替换)/logout(清除)/help(子命令帮助)。

update/logout 与 emcli/wxpub 同款语义——update 只管替换(新 key 从管理端复制,
获取凭据走 login 设备流),logout 只清本地(保留 base_url,三家共用文件的影响要说清)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from earshot.auth import clear_credentials
from earshot.cli import main
from earshot.config import Settings


def _settings() -> Settings:
    return Settings(
        home=Path("/tmp/earshot-test"),
        model_root=Path("/tmp/earshot-test/models"),
        exomind_base_url="https://exomind.example",
        exomind_api_key="",
    )


def _write_config(path: Path, api_key: str = "sk_live_old_key") -> None:
    path.write_text(json.dumps({"base_url": "https://exomind.example", "api_key": api_key}), encoding="utf-8")


class TestClearCredentials:
    def test_清除_api_key_保留_base_url(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.json"
        _write_config(cfg)
        assert clear_credentials(cfg) == cfg
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data["api_key"] == ""
        assert data["base_url"] == "https://exomind.example"

    def test_无配置文件或本就无凭据_返回_none(self, tmp_path: Path) -> None:
        assert clear_credentials(tmp_path / "absent.json") is None
        empty = tmp_path / "empty.json"
        _write_config(empty, api_key="")
        assert clear_credentials(empty) is None


class TestCmdUpdate:
    def test_新_key_探活通过_替换落盘(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr("earshot.cli.verify_key", lambda base, key: True)
        saved: dict[str, str] = {}

        def fake_save(base_url: str, api_key: str) -> Path:
            saved["base_url"], saved["key"] = base_url, api_key
            return tmp_path / "config.json"

        monkeypatch.setattr("earshot.cli.save_credentials", fake_save)
        args = argparse.Namespace(api_key="sk_live_new_key")
        from earshot.cli import cmd_update

        code = cmd_update(args, _settings())
        assert code == 0
        assert saved == {"base_url": "https://exomind.example", "key": "sk_live_new_key"}
        assert "凭据已替换" in capsys.readouterr().out

    def test_探活失败_不写入(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("earshot.cli.verify_key", lambda base, key: False)
        monkeypatch.setattr("earshot.cli.save_credentials", lambda *a, **k: pytest.fail("探活失败不得落盘"))
        args = argparse.Namespace(api_key="sk_live_bad")
        from earshot.cli import cmd_update

        code = cmd_update(args, _settings())
        assert code == 1
        assert "未写入" in capsys.readouterr().out


class TestCmdLogout:
    def test_有凭据_清除并提示共用影响(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        monkeypatch.setattr("earshot.cli.clear_credentials", lambda: tmp_path / "config.json")
        from earshot.cli import cmd_logout

        code = cmd_logout(argparse.Namespace(), _settings())
        out = capsys.readouterr().out
        assert code == 0
        assert "已登出" in out
        assert "fireCut" in out, "三家共用 config.json 的影响必须说清"

    def test_无凭据_无事可做(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("earshot.cli.clear_credentials", lambda: None)
        from earshot.cli import cmd_logout

        code = cmd_logout(argparse.Namespace(), _settings())
        assert code == 0
        assert "无需登出" in capsys.readouterr().out


class TestHelpCommand:
    def test_help_无参数_打印总览(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["help"]) == 0
        out = capsys.readouterr().out
        assert "usage: earshot" in out
        assert "update" in out and "logout" in out, "新命令要在帮助里可见"

    def test_help_带命令名_打印该命令帮助(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["help", "login"]) == 0
        assert "--force" in capsys.readouterr().out

    def test_help_未知命令_报错不崩(self, capsys: pytest.CaptureFixture[str]) -> None:
        # argparse 对无效 choice 打错误 + 退出码 2,main 收敛为返回值
        assert main(["help", "no-such-cmd"]) == 2
