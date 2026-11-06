"""测试公共桩:headless CI(Linux 无音频栈)上 stub 掉 soundcard。

soundcard 在无 PulseAudio 守护的环境 import 即挂(OSError: libpulse 缺失 /
AssertionError: 无守护),而 capture.py 仅在函数内使用它——stub 成「调用即报错」
的空模块。采集类测试本就不在 CI 跑,投递/向导/登录等纯逻辑测试不被 import 连坐。
"""

from __future__ import annotations

import sys
import types

try:
    import soundcard  # noqa: F401
except Exception:  # OSError / AssertionError 均视为不可用
    stub = types.ModuleType("soundcard")

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("soundcard 不可用(headless 环境 stub)")

    for _name in (
        "all_microphones",
        "default_speaker",
        "default_microphone",
        "get_microphone",
        "get_speaker",
    ):
        setattr(stub, _name, _unavailable)
    sys.modules["soundcard"] = stub
