"""config 模块单测:优先级与环境变量覆盖。"""

import json
from pathlib import Path

from earshot.config import (
    DEFAULT_EXOMIND_BASE_URL,
    load_exomind_config,
    load_settings,
)


def test_load_settings_defaults_from_env(tmp_path: Path) -> None:
    env = {"EARSHOT_HOME": str(tmp_path)}
    s = load_settings(env=env, exomind_config_path=tmp_path / "none.json")
    assert s.home == tmp_path
    assert s.model_root == tmp_path / "models"
    assert s.outbox == tmp_path / "outbox"
    assert s.exomind_base_url == DEFAULT_EXOMIND_BASE_URL
    assert s.exomind_api_key == ""


def test_load_settings_env_beats_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps({"base_url": "https://file.example.com/", "api_key": "from-file"}),
        encoding="utf-8",
    )
    env = {
        "EARSHOT_HOME": str(tmp_path),
        "EARSHOT_MODEL_ROOT": str(tmp_path / "m"),
        "EXOMIND_BASE_URL": "https://env.example.com/",
        "EXOMIND_API_KEY": "from-env",
    }
    s = load_settings(env=env, exomind_config_path=cfg)
    assert s.model_root == tmp_path / "m"
    assert s.exomind_base_url == "https://env.example.com"  # env 优先 + 去尾斜杠
    assert s.exomind_api_key == "from-env"


def test_load_settings_reads_exomind_config(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"base_url": "https://file.example.com", "api_key": "k1"}), encoding="utf-8")
    s = load_settings(env={"EARSHOT_HOME": str(tmp_path)}, exomind_config_path=cfg)
    assert s.exomind_base_url == "https://file.example.com"
    assert s.exomind_api_key == "k1"


def test_load_exomind_config_tolerates_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_exomind_config(bad) == {}
    assert load_exomind_config(tmp_path / "missing.json") == {}


def test_load_exomind_config_drops_non_string_values(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"api_key": "ok", "port": 8080, "nested": {"a": 1}}), encoding="utf-8")
    assert load_exomind_config(cfg) == {"api_key": "ok"}
