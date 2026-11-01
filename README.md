# earshot

旁听剪藏:**听到的,都能进知识库。**

听 B站/视频号/播客/讲座时开着它,听完 `Ctrl-C`,录音自动转写为 Markdown 并投递到你的知识库(默认 [ExoMind 知识飞轮](https://exomind.ai-as.cn),可插拔任意 webhook)。纯端侧转写,无限流、零 API 费用。

```
系统声音 ──录制──> 16k wav ──silero VAD 切段──> SenseVoice 端侧识别(实时字幕) ──> Markdown ──> 投递(exomind / webhook / 仅存档)
视频链接 ──yt-dlp──> 平台字幕(ai-zh/CC) ────────────────────────────┘(高信噪比路线)
```

## 安装(开箱即用)

```bash
uv tool install earshot-cli   # 或 pipx install earshot-cli
earshot doctor                # 环境自检;earshot 与 earshot-cli 两个命令名等价可用
```

源码安装请**锁版本 tag**,不要引滚动的 `main`(消费方 `uv sync`/`pip install` 会隐性升级到未审阅代码;升级应当是显式改版本号的决策):

```bash
uv tool install "earshot-cli @ https://github.com/helloworldtang/earshot/archive/refs/tags/v0.1.0.tar.gz"
```

- **零系统依赖**:ffmpeg 已随 Python wheel 内置([imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg),MoviePy 同款),无需 brew/apt。
- **模型首次运行自动下载**:SenseVoice int8 + silero VAD(~230MB,默认走 hf-mirror,`HF_ENDPOINT` 可覆盖),装完不再下。
- **投递免装 CLI**:默认直连 ExoMind HTTP API(复用 `~/.exomind/config.json` 凭据);没有凭据但有 exomind CLI 时自动回退;也支持任意 webhook。

### 平台采集矩阵

| 平台 | 系统声采集 | 额外步骤 |
|---|---|---|
| Windows | WASAPI loopback(原生) | 无 |
| Linux | PulseAudio/PipeWire `.monitor` | 无 |
| macOS | 无原生环回 | 见下方「macOS 三级台阶」,零配置即可用 |

#### macOS 三级台阶(按门槛从低到高)

1. **零配置·录屏路线**:听 Mac 上的内容时 `Cmd+Shift+5` 录屏(自带系统声),
   听完 `earshot audio 录屏.mp4` —— 不装任何东西,今天就能用。
2. **零配置·麦录兜底**:直接 `earshot listen`,未配置环回时自动退回内置麦克风
   外放收音(豆包同款,音质一般但可用)。
3. **一次性设置·内录(音质无损)**:`brew install blackhole-2ch`,在「音频 MIDI 设置」
   新建**多输出设备**(BlackHole 2ch + 扬声器)并设为系统输出 —— 边听边录,终身受用。
   装好后 earshot 自动选中,无需 `--device`。

> macOS 下默认逻辑 = 取「当前默认扬声器的同名输入设备」;把多输出设备设为默认输出即可被自动选中。也可 `--device` 指定。

## 使用

```bash
# 旁听模式:开录 → 听(实时字幕滚动) → Ctrl-C 完成 → 预览确认 → 投递
earshot listen
earshot listen --note "B站·xxx第3期"

# B站等视频链接直拉字幕(高信噪比:无环境噪音/打断;AI 字幕需登录态)
earshot video "https://www.bilibili.com/video/BVxxxx" --cookies-from-browser chrome

# 已有音视频文件(录屏 mp4 / 语音 m4a / 播客 mp3)
earshot audio recording.mp4 --title "通勤听书笔记" --deliver none

# 补投/重投 outbox 里的笔记(如环境问题当时没投出去)
earshot deliver ~/.earshot/outbox/20261030-xxxx.md

# 列出采集设备 / 环境自检 / 清理过期留档
earshot devices
earshot doctor
earshot clean --days 30
```

常用参数:`--title`(默认取首句截断40字)、`--tag`(可多次)、`--note`(来源备注写入脚注)、
`--engine server`(改走云端 dashscope 转写,复用 `~/.exomind/config.json` 凭据)、
`--deliver webhook --webhook <url>`(投递到任意知识库,POST JSON `{origin,title,content,tags}`)、
`--deliver none`(只落 `~/.earshot/outbox/` 存档)。

笔记留档:`~/.earshot/outbox/YYYYMMDD-HHMMSS-<标题>.md`。

### 文件生命周期

| 位置 | 内容 | 生命周期 |
|---|---|---|
| `~/.earshot/tmp/*.wav` | 录音中间产物 | 转写投递成功后**自动删除**(失败保留供排查) |
| `~/.earshot/outbox/*.md` | 转写笔记留档 | **永久保留**(审计/重投兜底),`earshot clean --days 30` 手动清理 |

投递前会先在终端**预览识别内容**,回车投递、`q` 放弃(笔记仍留档);`--yes` 跳过确认。等待服务端回执时 Ctrl-C = 不再等待(内容已提交,job 在服务端继续)。

## 设计要点

- **声音是最后的通用接口**:不依赖任何平台的开放性(B站字幕 API、视频号加密……都不碰),换平台零成本。
- **端侧优先**:SenseVoice-small int8 纯 CPU 转写,1 小时音频约 1-2 分钟,无"服务用户多请稍后"。
- **投递可插拔**:exomind CLI(复用其异步 job、轮询、client_key 幂等)/ webhook / 仅存档。
- **平台差异只允许存在于 `capture.py`**:换采集方案不动上层。

## 开发

```bash
uv sync                      # 含 dev 组(ruff/mypy/pytest)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

测试不依赖模型与声卡;`tests/test_extract.py` 需要本机 ffmpeg(无则自动跳过)。

## License

MIT
