"""earshot:旁听剪藏 —— 听到的都能进知识库。

跨平台桌面端:系统声环回录制(macOS 需 BlackHole)→ 端侧 VAD+ASR 转写
→ 组装 Markdown → 投递(默认 exomind CLI,可插拔 webhook)。
"""

from importlib import metadata as _metadata

try:
    # 单一版本源:pyproject.toml(已安装发行版的元数据);源码 checkout 未安装时回退。
    # 旧写法在此手写 "0.1.1",与 pyproject 的 0.3.x 漂移两月无人察觉——UA 带版本
    # 后漂移会直接暴露在服务端日志里,先根治版本源。
    # 注意发行版名是 earshot-cli(import 名才是 earshot),两个键都试。
    __version__ = _metadata.version("earshot-cli")
except _metadata.PackageNotFoundError:  # pragma: no cover - 仅源码运行路径
    __version__ = "0.0.dev0"


def ua_headers(**extra: str) -> dict[str, str]:
    """自有服务(exomind)请求头:显式 UA(可观测性标配)+ 附加头(Authorization 等)。

    全仓唯一注入点——deliver/auth/transcribe_cloud 旁路端点一律复用,不写两遍;
    第三方 webhook(deliver_webhook)不经过这里(无自有服务端可分桶,不值得改)。
    """
    return {"User-Agent": f"earshot/{__version__}", **extra}
