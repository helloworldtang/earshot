"""配置加载:earshot 数据目录、模型根目录、exomind 凭据。

优先级:环境变量 > ``~/.exomind/config.json`` > 默认值。
exomind 凭据与 exomind CLI 共用(``~/.exomind/config.json``),云引擎与投递不用重复登录。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

EARSHOT_HOME_ENV = "EARSHOT_HOME"
EARSHOT_MODEL_ROOT_ENV = "EARSHOT_MODEL_ROOT"
EXOMIND_API_KEY_ENV = "EXOMIND_API_KEY"
EXOMIND_BASE_URL_ENV = "EXOMIND_BASE_URL"

DEFAULT_EXOMIND_BASE_URL = "https://exomind-api.ai-as.cn"
EXOMIND_CONFIG_FILENAME = ".exomind/config.json"


@dataclass(frozen=True, slots=True)
class Settings:
    """运行时配置快照。

    Attributes:
        home: earshot 数据目录(outbox 留档、tmp 录音)。
        model_root: 模型根目录,内含 ``asr/`` 与 ``vad/``。
        exomind_base_url: exomind API 地址(云引擎 + 投递)。
        exomind_api_key: API key,未配置时为空串(仅本地模式可用)。
    """

    home: Path
    model_root: Path
    exomind_base_url: str
    exomind_api_key: str

    @property
    def outbox(self) -> Path:
        """转写笔记留档目录。"""
        return self.home / "outbox"


def load_settings(
    env: dict[str, str] | None = None,
    exomind_config_path: Path | None = None,
) -> Settings:
    """解析配置。

    Args:
        env: 环境变量映射,默认取 ``os.environ``(测试注入用)。
        exomind_config_path: exomind 配置文件路径,默认 ``~/.exomind/config.json``。

    Returns:
        Settings 快照。
    """
    env = dict(os.environ) if env is None else env
    exo = load_exomind_config(exomind_config_path)
    home = Path(env.get(EARSHOT_HOME_ENV, str(Path.home() / ".earshot")))
    model_root = Path(env.get(EARSHOT_MODEL_ROOT_ENV, str(home / "models")))
    base_url = env.get(EXOMIND_BASE_URL_ENV) or exo.get("base_url") or DEFAULT_EXOMIND_BASE_URL
    api_key = env.get(EXOMIND_API_KEY_ENV) or exo.get("api_key") or ""
    return Settings(
        home=home,
        model_root=model_root,
        exomind_base_url=base_url.rstrip("/"),
        exomind_api_key=api_key,
    )


def load_exomind_config(path: Path | None = None) -> dict[str, str]:
    """读取 exomind CLI 的配置文件,不存在或损坏时返回空 dict。

    Args:
        path: 配置文件路径,默认 ``~/.exomind/config.json``。

    Returns:
        只含 ``base_url`` / ``api_key`` 等字符串键的 dict。
    """
    cfg_path = path if path is not None else Path.home() / EXOMIND_CONFIG_FILENAME
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}
