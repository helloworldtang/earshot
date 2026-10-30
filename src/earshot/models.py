"""模型自动获取:缺失时首次运行自动下载(MacWhisper/Ollama 式体验)。

源默认走 hf-mirror(国内可达),可用 HF_ENDPOINT 覆盖。
"""

from __future__ import annotations

from pathlib import Path

import httpx

DEFAULT_ENDPOINT = "https://hf-mirror.com"
SENSE_VOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SILERO_VAD_REPO = "onnx-community/silero-vad"

# 相对路径 -> (HF 仓库, 仓库内文件)
MODEL_FILES: dict[str, tuple[str, str]] = {
    "asr/model.int8.onnx": (SENSE_VOICE_REPO, "model.int8.onnx"),
    "asr/tokens.txt": (SENSE_VOICE_REPO, "tokens.txt"),
    "vad/silero_vad.onnx": (SILERO_VAD_REPO, "onnx/model.onnx"),
}


def missing_models(model_root: Path) -> list[str]:
    """列出缺失的模型相对路径。"""
    return [rel for rel in MODEL_FILES if not (model_root / rel).is_file()]


def ensure_models(model_root: Path, endpoint: str | None = None) -> bool:
    """确保模型齐全,缺失则下载。

    Args:
        model_root: 模型根目录。
        endpoint: HF 端点,默认 hf-mirror,可 ``HF_ENDPOINT`` 覆盖。

    Returns:
        是否发生了下载。

    Raises:
        RuntimeError: 下载失败(网络不可达等)。
    """
    import os

    base = endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    downloaded = False
    for rel in missing_models(model_root):
        repo, remote = MODEL_FILES[rel]
        dest = model_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{repo}/resolve/main/{remote}"
        print(f"↓ 下载模型 {rel}(首次运行自动补齐,之后不再下载)")
        try:
            mb = 1 << 20
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or 0)
                done = 0
                with dest.with_suffix(dest.suffix + ".part").open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=mb):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done // mb}/{total // mb} MB", end="", flush=True)
            dest.with_suffix(dest.suffix + ".part").replace(dest)
            downloaded = True
            print()
        except httpx.HTTPError as e:
            for suffix in ("", ".part"):
                Path(str(dest) + suffix).unlink(missing_ok=True)
            msg = f"模型下载失败: {rel}({e})。可设 HF_ENDPOINT=https://huggingface.co 重试,或手动放置到 {model_root}"
            raise RuntimeError(msg) from e
    return downloaded
