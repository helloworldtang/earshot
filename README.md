# earshot

旁听剪藏:**听到的,都能进知识库。**

听 B站/视频号/播客/讲座时开着它,听完 `Ctrl-C`,录音自动转写为 Markdown 并投递到你的知识库(默认 [ExoMind 知识飞轮](https://exomind.ai-as.cn),可插拔任意 webhook)。纯端侧转写,无限流、零 API 费用。

```
系统声音 ──录制──> 16k wav ──silero VAD 切段──> SenseVoice 端侧识别(实时字幕) ──> Markdown ──> 投递(exomind / webhook / 仅存档)
视频链接 ──yt-dlp──> 平台字幕(ai-zh/CC) ────────────────────────────┘(高信噪比路线)
```

## 安装(开箱即用)

唯一的依赖是 [uv](https://docs.astral.sh/uv/)(Python 的包管理器)。**不需要预装 Python**——uv 会自动下载并管理所需版本(earshot 需要 Python ≥3.12,uv 全程代劳)。

按顺序三步:

**第 1 步:安装 uv**(已装可跳过,`uv --version` 可验证)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows(PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

装完**重开一个终端**(或按安装器提示把 uv 加进 PATH),再验证:

```bash
uv --version
```

<details>
<summary><b>国内网络看这里</b>(直连 GitHub/PyPI 慢或超时)</summary>

uv 官方安装脚本与 Python 下载默认走 GitHub、包索引走 PyPI 官方,大陆网络可能超时。两条路任选:

**路线 A(推荐,一条命令全搞定)**:[uv.agentsmirror.com](https://github.com/Wangnov/uv-custom) 公益镜像——分发的是官方原版安装脚本,只把下载源切到国内,并自动写入 Python 下载镜像与 PyPI 镜像配置:

```bash
# macOS / Linux
curl -LsSf https://uv.agentsmirror.com/install-cn.sh | sh
# Windows(PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://uv.agentsmirror.com/install-cn.ps1 | iex"
```

装完直接跳到第 2 步,后续 `uv tool install` 的包下载也已走国内源。

**路线 B(已装 Python 的用户)**:用 pip 从清华源装 uv:

```bash
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

再给 uv 配国内 PyPI 源(一劳永逸,写入 `~/.config/uv/uv.toml`):

```bash
printf '[[index]]\nurl = "https://pypi.tuna.tsinghua.edu.cn/simple"\ndefault = true\n' > ~/.config/uv/uv.toml
```

> 路线 B 未覆盖「uv 自动下载 Python」这一步(仍走 GitHub):若第 2 步卡在下载 Python 且本机 Python 版本低于 3.12,改走路线 A(它会一并配好 Python 下载镜像)。

</details>

**第 2 步:安装 earshot**(uv 自动下载 Python ≥3.12 与全部依赖,含内置 ffmpeg)

```bash
uv tool install earshot-cli
```

> 若提示 `command not found: earshot`:uv 把命令装到了 `~/.local/bin`(Windows 为 `%USERPROFILE%\.local\bin`),按提示把它加入 PATH 或重开终端。

**第 3 步:登录并自检**

```bash
earshot login    # 浏览器打开授权页,点「授权登录」即完成(已登录过 exomind 则免)
earshot me       # 显示凭据状态与服务连通性(「凭据状态:有效」即就绪)
```

没有 ExoMind 账号?投递可换任意 webhook(`--deliver webhook --webhook <url>`),或 `--deliver none` 只本地存档,两者都不需要登录。

`earshot` 与 `earshot-cli` 两个命令名等价可用;`earshot doctor` 可做完整环境自检。

升级:

```bash
uv tool upgrade earshot-cli
# 刚发布的版本若报 "Nothing to upgrade":PyPI 索引传播延迟(约 1 分钟),
# 稍后再试,或立即生效:uv tool install --force earshot-cli
```

源码安装请**锁版本 tag**,不要引滚动的 `main`(消费方 `uv sync`/`pip install` 会隐性升级到未审阅代码;升级应当是显式改版本号的决策):

```bash
uv tool install "earshot-cli @ https://github.com/helloworldtang/earshot/archive/refs/tags/v0.3.0.tar.gz"
```

- **零系统依赖**:ffmpeg 已随 Python wheel 内置([imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg),MoviePy 同款),无需 brew/apt。
- **模型首次运行自动下载**:SenseVoice int8 + silero VAD(~230MB,默认走 hf-mirror,`HF_ENDPOINT` 可覆盖),装完不再下。
- **投递免装 CLI**:默认直连 ExoMind HTTP API(复用 `~/.exomind/config.json` 凭据);没有凭据但有 exomind CLI 时自动回退;也支持任意 webhook。
- **独立登录免装 emcli**:`earshot login` 设备流浏览器授权,直接写入 `~/.exomind/config.json`(与 emcli 共用同一凭据文件);已装 emcli 且登录过则无需再登。

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

# 查看凭据状态 / 列出采集设备 / 环境自检 / 清理过期留档
earshot me
earshot devices
earshot doctor
earshot clean --days 30
```

常用参数:`--title`(默认取首句截断40字)、`--tag`(可多次)、`--note`(来源备注写入脚注)、
`--engine server`(改走云端 dashscope 转写,复用 `~/.exomind/config.json` 凭据)、
`--deliver webhook --webhook <url>`(投递到任意知识库,POST JSON `{origin,title,content,tags}`)、
`--deliver none`(只落 `~/.earshot/outbox/` 存档)。

首次使用取凭据:`earshot login`(浏览器设备流授权;凭据仍有效时直接返回,`--force` 强制重登),
或设 `EXOMIND_API_KEY` 环境变量(从 exomind 网页 `/ui/account` 复制)。

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
