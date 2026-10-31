"""earshot CLI:listen / audio / devices / doctor。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from earshot import __version__
from earshot.capture import blackhole_missing_hint, list_capture_devices, pick_device, record_to_wav
from earshot.compose import NoteMeta, derive_title, group_paragraphs, outbox_path, render_markdown
from earshot.config import Settings, load_settings
from earshot.deliver import DeliveryResult, deliver_exomind, deliver_exomind_api, deliver_webhook
from earshot.extract import ExtractError, ffmpeg_available, to_wav16k, wav_duration
from earshot.live import LiveSession
from earshot.models import ensure_models
from earshot.subtitle import SubtitleError, fetch_video_note
from earshot.transcribe import LocalTranscriber, Segment, TranscribeError
from earshot.transcribe_cloud import CloudTranscribeError, transcribe_cloud

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_CONTENT = 2

_LEVEL_BLOCKS = "▁▂▃▄▅▆▇█"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--title", help="笔记标题(默认取首句截断)")
    p.add_argument("--tag", action="append", default=None, help="投递标签,可多次(默认 旁听)")
    p.add_argument("--note", default="", help="来源备注,如「B站·xxx第3期」,写进脚注")
    p.add_argument("--engine", choices=["local", "server"], default="local", help="转写引擎(默认 local 端侧)")
    p.add_argument(
        "--deliver",
        choices=["exomind", "webhook", "none"],
        default="exomind",
        help="投递方式(默认 exomind CLI)",
    )
    p.add_argument("--webhook", help="webhook 地址(--deliver webhook 时必填)")
    p.add_argument("--out", type=Path, help="outbox 目录(默认 ~/.earshot/outbox)")
    p.add_argument("-y", "--yes", action="store_true", help="跳过内容确认,直接投递")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earshot", description="旁听剪藏:听到的都能进知识库")
    parser.add_argument("--version", action="version", version=f"earshot {__version__}")
    sub = parser.add_subparsers(dest="command")  # 不带子命令时进入交互向导

    p_listen = sub.add_parser("listen", help="录系统声音(Ctrl-C=完成),转写并投递")
    _add_common(p_listen)
    p_listen.add_argument("--device", help="采集设备名子串(默认:默认扬声器的环回)")

    p_audio = sub.add_parser("audio", help="转写已有音视频文件并投递")
    _add_common(p_audio)
    p_audio.add_argument("file", type=Path, nargs="?", help="音视频文件路径(缺省时交互询问)")

    p_video = sub.add_parser("video", help="B站等视频链接直拉字幕入库(无噪音路线)")
    _add_common(p_video)
    p_video.add_argument("url", nargs="?", help="视频链接(B站等 yt-dlp 支持的平台;缺省时交互询问)")
    p_video.add_argument(
        "--cookies-from-browser",
        help="浏览器名(chrome/safari/firefox/edge),拉 AI 字幕(ai-zh)通常需要登录态",
    )

    p_deliver = sub.add_parser("deliver", help="补投/重投 outbox 里的笔记")
    p_deliver.add_argument("file", type=Path, help="笔记 md 路径")
    p_deliver.add_argument("--title", help="标题(默认取首行 # 标题)")
    p_deliver.add_argument("--tag", action="append", default=None, help="标签,可多次(默认 旁听)")
    p_deliver.add_argument(
        "--deliver", choices=["exomind", "webhook", "none"], default="exomind", help="投递方式(默认 exomind)"
    )
    p_deliver.add_argument("--webhook", help="webhook 地址(--deliver webhook 时必填)")

    sub.add_parser("devices", help="列出可用采集设备")
    sub.add_parser("doctor", help="环境自检")
    p_clean = sub.add_parser("clean", help="清理过期留档(outbox+tmp)")
    p_clean.add_argument("--days", type=int, default=30, help="删除超过 N 天的文件(默认 30)")
    return parser


def _transcribe(wav16k: Path, engine: str, settings: Settings) -> list[Segment]:
    """按引擎转写,返回 Segment 列表(server 引擎按行造伪段)。"""
    if engine == "server":
        print("→ 云端转写中(经 exomind /transcribe,dashscope)…")
        text = transcribe_cloud(wav16k, settings)
        return [Segment(0.0, 0.0, line.strip()) for line in text.splitlines() if line.strip()]
    print("→ 本地转写中(VAD+SenseVoice)…")
    ensure_models(settings.model_root)  # 缺失则首次运行自动下载(hf-mirror)
    transcriber = LocalTranscriber(settings.model_root)

    def _progress(frac: float) -> None:
        print(f"\r→ 本地转写中… {frac * 100:3.0f}%", end="", flush=True)

    segments = transcriber.transcribe(wav16k, on_progress=_progress)
    print()
    return segments


def _run_pipeline(
    segments: list[Segment],
    meta: NoteMeta,
    args: argparse.Namespace,
    settings: Settings,
) -> int:
    """组装 md → 落 outbox → 投递,返回退出码。"""
    if not any(s.text.strip() for s in segments):
        print("未识别到语音内容(检查系统输出/设备,或音频本身无人声)")
        return EXIT_NO_CONTENT

    title = derive_title(segments, args.title)
    tags = args.tag if args.tag else ["旁听"]
    paragraphs = ["".join(s.text for s in group) for group in group_paragraphs(segments)]
    md = render_markdown(meta, paragraphs, title)

    outbox = args.out if args.out else settings.outbox
    outbox.mkdir(parents=True, exist_ok=True)
    md_path = outbox_path(outbox, meta.started_at, title)
    md_path.write_text(md, encoding="utf-8")
    print(f"已保存笔记: {md_path}")

    # 投递前先看内容:识别差可当场放弃(笔记仍留档)
    print("\n=== 识别内容 ===")
    for para in paragraphs:
        print(para + "\n")
    print("=== 内容结束 ===")
    if sys.stdin.isatty() and not args.yes:
        try:
            answer = input("[回车] 投递 | [q] 放弃投递(笔记保留): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n已中断;笔记保留在 outbox")
            return EXIT_OK
        if answer == "q":
            print("已放弃投递;笔记保留在 outbox,可稍后编辑后手动 ingest")
            return EXIT_OK

    if args.deliver == "none":
        print("跳过投递(--deliver none)")
        return EXIT_OK
    return _do_deliver(title, md, md_path, tags, args.deliver, args.webhook, settings)


def _do_deliver(
    title: str,
    md: str,
    md_path: Path,
    tags: list[str],
    deliver_kind: str,
    webhook: str | None,
    settings: Settings,
) -> int:
    """执行投递并汇报,返回退出码(webhook / exomind API / exomind CLI)。"""
    if deliver_kind == "none":
        print("跳过投递(--deliver none)")
        return EXIT_OK
    result: DeliveryResult | None = None
    try:
        if deliver_kind == "webhook":
            if not webhook:
                print("✗ --deliver webhook 需要 --webhook <url>")
                return EXIT_FAIL
            result = deliver_webhook(webhook, title, md, tags)
        elif settings.exomind_api_key:
            result = deliver_exomind_api(md_path, title, tags, settings)  # 直连 API,免装 CLI
        elif shutil.which("exomind"):
            result = deliver_exomind(md_path, title, tags)  # 无凭据但有 CLI,回退
    except KeyboardInterrupt:
        print("\n⏳ 已中断:内容可能已提交(服务端会继续处理);笔记保留在 outbox")
        return EXIT_OK
    if result is None:
        cfg_path = Path.home() / ".exomind" / "config.json"
        print("✗ failed: 无 exomind 凭据也无 exomind CLI,自诊断:")
        print(f"  HOME={Path.home()}")
        print(f"  配置文件 {cfg_path} 存在={cfg_path.is_file()}")
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                print(f"  api_key 字段长度={len(str(cfg.get('api_key') or ''))}")
            except ValueError as e:
                print(f"  配置文件解析失败: {e}")
        print(f"  PATH={os.environ.get('PATH', '')[:200]}")
        print("  → exomind login 生成 ~/.exomind/config.json,或 npm i -g exomind,或改用 --deliver webhook/none")
        print("  → 笔记已留档,环境修好后可用 earshot deliver <md路径> 补投")
        return EXIT_FAIL
    mark = {"delivered": "✓", "pending": "⏳", "failed": "✗"}.get(result.status, "·")
    print(f"{mark} {result.status}: {result.detail}")
    return EXIT_OK if result.status in ("delivered", "pending") else EXIT_FAIL


def cmd_listen(args: argparse.Namespace, settings: Settings) -> int:
    try:
        mic, mode_label = pick_device(args.device)
    except LookupError as e:
        print(f"✗ {e}")
        return EXIT_FAIL
    print(f"采集: {mode_label}")

    tmp_dir = settings.home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    raw_wav = tmp_dir / f"listen-{started_at.strftime('%Y%m%d-%H%M%S')}.wav"

    # 实时字幕:本地引擎边录边转,段闭合即上屏
    live: LiveSession | None = None
    on_audio = None
    if args.engine == "local":
        try:
            ensure_models(settings.model_root)
            transcriber = LocalTranscriber(settings.model_root)
        except (RuntimeError, TranscribeError) as e:
            print(f"✗ {e}")
            return EXIT_FAIL

        def _show(text: str) -> None:
            print(f"\r\033[K  ↪ {text}", flush=True)  # 清掉电平行再出字幕

        live = LiveSession(transcriber, on_text=_show)
        live.start()

        import soxr

        resampler = soxr.ResampleStream(48000, 16000, num_channels=1)

        def _push_16k(mono: np.ndarray) -> None:
            assert live is not None
            live.push(np.asarray(resampler.resample_chunk(mono), dtype=np.float32))

        on_audio = _push_16k

    print("录音中… 听完按 Ctrl-C 结束(实时字幕随说话滚动;无字幕/电平不动=检查麦克风权限)")
    import time as _time

    t0 = _time.monotonic()

    def _meter(rms: float) -> None:
        elapsed = _time.monotonic() - t0
        m, s = divmod(int(elapsed), 60)
        level = int(min(1.0, max(rms, 0.0) * 8.0) * len(_LEVEL_BLOCKS))
        bar = _LEVEL_BLOCKS[level - 1] * max(level, 1) if level else "·"
        print(f"\r⏺ {m:02d}:{s:02d} {bar:<8}", end="", flush=True)

    duration_s = record_to_wav(mic, raw_wav, on_level=_meter, on_audio=on_audio)
    print()
    if duration_s < 1.0:
        print(f"录音太短({duration_s:.1f}s),忽略")
        return EXIT_NO_CONTENT

    if live is not None:
        print("→ 收尾转写中(最后未闭段)…")
        try:
            segments = live.finalize()
        except TranscribeError as e:
            print(f"✗ {e}")
            return EXIT_FAIL
    else:
        wav16k = raw_wav.with_suffix(".16k.wav")
        try:
            to_wav16k(raw_wav, wav16k)
        except ExtractError as e:
            print(f"✗ {e}")
            return EXIT_FAIL
        try:
            segments = _transcribe(wav16k, args.engine, settings)
        except (TranscribeError, CloudTranscribeError) as e:
            print(f"✗ {e}")
            return EXIT_FAIL

    meta = NoteMeta(started_at=started_at, duration_s=duration_s, device=mic.name, source="listen", note=args.note)
    rc = _run_pipeline(segments, meta, args, settings)
    if rc == EXIT_OK:  # 成功后清掉临时录音(失败保留供排查)
        raw_wav.unlink(missing_ok=True)
        raw_wav.with_suffix(".16k.wav").unlink(missing_ok=True)
    return rc


def cmd_audio(args: argparse.Namespace, settings: Settings) -> int:
    src = args.file
    if src is None:
        if not _interactive():
            print("✗ 缺少文件路径:earshot audio <file>(或交互模式直接 earshot)")
            return EXIT_FAIL
        raw = _prompt("音视频文件路径(录屏 mp4 / 语音 m4a / 播客 mp3)")
        if not raw:
            print("✗ 未提供文件路径")
            return EXIT_FAIL
        src = Path(raw).expanduser()
    if not src.is_file():
        print(f"✗ 文件不存在: {src}")
        return EXIT_FAIL
    tmp_dir = settings.home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wav16k = tmp_dir / f"{src.stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.16k.wav"
    try:
        to_wav16k(src, wav16k)
        duration_s = wav_duration(wav16k)
    except ExtractError as e:
        print(f"✗ {e}")
        return EXIT_FAIL
    try:
        segments = _transcribe(wav16k, args.engine, settings)
    except (TranscribeError, CloudTranscribeError) as e:
        print(f"✗ {e}")
        return EXIT_FAIL
    meta = NoteMeta(
        started_at=datetime.fromtimestamp(src.stat().st_mtime),
        duration_s=duration_s,
        device="-",
        source="audio",
        note=args.note,
    )
    rc = _run_pipeline(segments, meta, args, settings)
    if rc == EXIT_OK:  # 成功后清掉临时 wav(失败保留供排查)
        wav16k.unlink(missing_ok=True)
    return rc


def _prompt(label: str, default: str = "", default_display: str | None = None) -> str:
    """交互询问一个字段:回车取默认值。

    Raises:
        KeyboardInterrupt / EOFError: 交由 main 统一优雅收场。
    """
    shown = default_display if default_display is not None else default
    suffix = f"(回车={shown})" if shown else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw or default


def _clipboard_text() -> str:
    """跨平台读剪贴板文本,不可用时返回空串。"""
    candidates = (
        ["pbpaste"],  # macOS
        ["xclip", "-selection", "clipboard", "-o"],  # Linux(X11)
        ["xsel", "--clipboard", "--output"],
        ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],  # Windows
    )
    import subprocess

    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)  # noqa: S603
        except Exception:
            continue
        if proc.returncode == 0:
            return proc.stdout.strip()
    return ""


def _interactive() -> bool:
    return bool(sys.stdin.isatty())


def cmd_video(args: argparse.Namespace, settings: Settings) -> int:
    url = args.url
    if not url:
        if not _interactive():
            print("✗ 缺少视频链接:earshot video <url>(或交互模式直接 earshot)")
            return EXIT_FAIL
        clip = _clipboard_text()
        if clip.lower().startswith("http"):
            url = _prompt("视频链接", default=clip, default_display="剪贴板")
        else:
            url = _prompt("视频链接")
        if not url:
            print("✗ 未提供链接")
            return EXIT_FAIL
    cookies = args.cookies_from_browser
    if cookies is None and _interactive():
        ans = _prompt("字幕登录态浏览器(拉 AI 字幕用;回车=chrome,输 n 匿名)", default="chrome")
        cookies = None if ans.lower() == "n" else ans
    try:
        meta, segments = fetch_video_note(url, cookies)
    except SubtitleError as e:
        print(f"✗ {e}")
        return EXIT_FAIL
    print(f"→ {meta.title}(UP:{meta.uploader},字幕:{meta.subtitle_lang},{len(segments)} 段)")
    if not args.title:
        args.title = meta.title  # 视频有正标题,优先于首句截断
    if not args.note:
        args.note = url  # 脚注默认放原文链接,实体页可回跳
    note_meta = NoteMeta(
        started_at=datetime.now(),
        duration_s=meta.duration_s,
        device=meta.uploader,
        source="video",
        note=args.note,
    )
    return _run_pipeline(segments, note_meta, args, settings)


def cmd_deliver(args: argparse.Namespace, settings: Settings) -> int:
    md_path = args.file
    if not md_path.is_file():
        print(f"✗ 文件不存在: {md_path}")
        return EXIT_FAIL
    md = md_path.read_text(encoding="utf-8")
    title = args.title
    if not title:
        first = next((ln[2:].strip() for ln in md.splitlines() if ln.startswith("# ")), "")
        title = first or md_path.stem
    tags = args.tag if args.tag else ["旁听"]
    if args.deliver == "none":
        print("跳过投递(--deliver none)")
        return EXIT_OK
    return _do_deliver(title, md, md_path, tags, args.deliver, args.webhook, settings)


def cmd_devices(_args: argparse.Namespace, _settings: Settings) -> int:
    devices = list_capture_devices()
    if not devices:
        print("未发现任何输入设备")
        return EXIT_FAIL
    print("输入设备(★=环回):")
    for name, is_loopback in devices:
        mark = "★" if is_loopback else " "
        print(f" {mark} {name}")
    hint = blackhole_missing_hint()
    if hint:
        print(f"\n提示: {hint}")
    return EXIT_OK


def cmd_doctor(_args: argparse.Namespace, settings: Settings) -> int:
    ok = True

    def check(label: str, fine: bool, fix: str = "") -> None:
        nonlocal ok
        ok = ok and fine
        print(f"{'✓' if fine else '✗'} {label}" + ("" if fine else f"  → {fix}"))

    check("ffmpeg(系统或内置)", ffmpeg_available(), "重装 earshot 依赖(imageio-ffmpeg 应自带)")
    try:
        devices = list_capture_devices()
        loopbacks = [n for n, lb in devices if lb]
        mac_hint = blackhole_missing_hint()
        if loopbacks:
            print(f"✓ 采集设备({len(loopbacks)} 个环回)")
        elif mac_hint is not None:
            # 麦录兜底存在,不算失败:降级为提示
            print(f"○ 采集设备:无环回,listen 将退回麦克风收音。{mac_hint}")
        else:
            check("采集设备", False, "无环回设备,检查系统音频")
    except Exception as e:  # soundcard 底层异常族不稳定,收口
        check("采集设备", False, f"soundcard 枚举失败: {e}")
    from earshot.models import missing_models

    missing = missing_models(settings.model_root)
    if missing:
        print(f"○ 模型:{len(missing)} 个缺失,首次转写时自动下载(~230MB,hf-mirror)")
    else:
        print("✓ 模型(SenseVoice int8 + silero VAD)")
    check(
        "exomind 投递(API 凭据或 CLI)",
        bool(settings.exomind_api_key) or shutil.which("exomind") is not None,
        "exomind login 生成 ~/.exomind/config.json,或 npm i -g exomind,或 --deliver webhook/none",
    )
    print("\n全部通过 ✓" if ok else "\n存在未通过项 ✗")
    return EXIT_OK if ok else EXIT_FAIL


def cmd_clean(args: argparse.Namespace, settings: Settings) -> int:
    """清理超过 N 天的 outbox 笔记与 tmp 录音。"""
    import time as _time

    cutoff = _time.time() - args.days * 86400
    removed = 0
    for folder in (settings.outbox, settings.home / "tmp"):
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
    print(f"已清理 {removed} 个超过 {args.days} 天的文件(outbox + tmp)")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    handlers = {
        "listen": cmd_listen,
        "audio": cmd_audio,
        "video": cmd_video,
        "deliver": cmd_deliver,
        "devices": cmd_devices,
        "doctor": cmd_doctor,
        "clean": cmd_clean,
    }
    if args.command is None:
        if not _interactive():
            parser.print_help()
            return EXIT_FAIL
        choice = _prompt("earshot 模式 [1]听录 [2]视频字幕 [3]本地文件(回车=1;输入 q 退出)", default="1")
        if choice == "q":
            return EXIT_OK
        mapping = {"1": "listen", "2": "video", "3": "audio"}
        sub = mapping.get(choice)
        if sub is None:
            print(f"✗ 无效选择: {choice}")
            return EXIT_FAIL
        args = parser.parse_args([sub])  # 取该子命令的默认值
    try:
        return int(handlers[args.command](args, settings))
    except (KeyboardInterrupt, EOFError):
        # Ctrl-C 语义 =「点完成」或「不等了」,不该以 traceback 收场
        print("\n(已中断)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
