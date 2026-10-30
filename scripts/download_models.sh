#!/usr/bin/env bash
# 拉取 earshot 本地转写模型(int8 优先,不进仓)。
#
# 产物布局(earshot 默认读 ~/.earshot/models,可用 EARSHOT_MODEL_ROOT 覆盖):
#   models/asr/model.int8.onnx   SenseVoice-small int8(zh/en/ja/ko/yue,含标点+ITN)
#   models/asr/tokens.txt
#   models/vad/silero_vad.onnx
#
# 源:HF(默认走 hf-mirror,本机实测 GitHub release 直连不通)。
# 用法: ./scripts/download_models.sh [目标目录,默认 ~/.earshot/models]
#       可 HF_ENDPOINT=https://huggingface.co 覆盖
set -euo pipefail

HF="${HF_ENDPOINT:-https://hf-mirror.com}"
ROOT="${1:-$HOME/.earshot/models}"

ASR_REPO="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
VAD_REPO="onnx-community/silero-vad"

fetch() {  # fetch <repo> <remote-file> <dest-relpath>
  local repo="$1" remote="$2" dest="$ROOT/$3"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then echo "skip  $3 (已存在)"; return; fi
  echo "get   $3"
  curl -L --fail --retry 3 -o "$dest.part" "$HF/$repo/resolve/main/$remote"
  mv "$dest.part" "$dest"
}

# ASR:SenseVoice int8(model ~230MB + tokens)
fetch "$ASR_REPO" model.int8.onnx asr/model.int8.onnx
fetch "$ASR_REPO" tokens.txt       asr/tokens.txt

# VAD:silero
fetch "$VAD_REPO" onnx/model.onnx  vad/silero_vad.onnx

echo "done: $ROOT"
du -sh "$ROOT"/* 2>/dev/null || true
