"""earshot CLI:listen / audio / devices / doctor。"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from earshot import __version__
from earshot.capture import blackhole_missing_hint, list_capture_devices, pick_device, record_to_wav
from earshot.compose import NoteMeta, derive_title, group_paragraphs, outbox_path, render_markdown
from earshot.config import Settings, load_settings
from earshot.deliver import DeliveryResult, deliver_exomind, deliver_exomind_api, deliver_webhook
from earshot.extract import ExtractError, ffmpeg_available, to_wav16k, wav_duration
from earshot.models import ensure_models
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earshot", description="旁听剪藏:听到的都能进知识库")
    parser.add_argument("--version", action="version", version=f"earshot {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_listen = sub.add_parser("listen", help="录系统声音(Ctrl-C=完成),转写并投递")
    _add_common(p_listen)
    p_listen.add_argument("--device", help="采集设备名子串(默认:默认扬声器的环回)")

    p_audio = sub.add_parser("audio", help="转写已有音视频文件并投递")
    _add_common(p_audio)
    p_audio.add_argument("file", type=Path, help="音视频文件路径")

    sub.add_parser("devices", help="列出可用采集设备")
    sub.add_parser("doctor", help="环境自检")
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

    if args.deliver == "none":
        print("跳过投递(--deliver none)")
        return EXIT_OK
    result: DeliveryResult
    if args.deliver == "webhook":
        if not args.webhook:
            print("✗ --deliver webhook 需要 --webhook <url>")
            return EXIT_FAIL
        result = deliver_webhook(args.webhook, title, md, tags)
    elif settings.exomind_api_key:
        result = deliver_exomind_api(md_path, title, tags, settings)  # 直连 API,免装 CLI
    elif shutil.which("exomind"):
        result = deliver_exomind(md_path, title, tags)  # 无凭据但有 CLI,回退
    else:
        print("✗ failed: 无 exomind 凭据也无 exomind CLI")
        print("  → exomind login 生成 ~/.exomind/config.json,或 npm i -g exomind,或改用 --deliver webhook/none")
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
    print("录音中… 听完按 Ctrl-C 结束(退出码即转写投递)")
    print("(若始终无电平条:检查 系统设置→隐私与安全性→麦克风 是否已授权本终端)")

    tmp_dir = settings.home / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    raw_wav = tmp_dir / f"listen-{started_at.strftime('%Y%m%d-%H%M%S')}.wav"

    import time as _time

    t0 = _time.monotonic()

    def _meter(rms: float) -> None:
        elapsed = _time.monotonic() - t0
        m, s = divmod(int(elapsed), 60)
        level = int(min(1.0, max(rms, 0.0) * 8.0) * len(_LEVEL_BLOCKS))
        bar = _LEVEL_BLOCKS[level - 1] * max(level, 1) if level else "·"
        print(f"\r⏺ {m:02d}:{s:02d} {bar:<8}", end="", flush=True)

    duration_s = record_to_wav(mic, raw_wav, on_level=_meter)
    print()
    if duration_s < 1.0:
        print(f"录音太短({duration_s:.1f}s),忽略")
        return EXIT_NO_CONTENT

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
    return _run_pipeline(segments, meta, args, settings)


def cmd_audio(args: argparse.Namespace, settings: Settings) -> int:
    src = args.file
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
    return _run_pipeline(segments, meta, args, settings)


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    handlers = {
        "listen": cmd_listen,
        "audio": cmd_audio,
        "devices": cmd_devices,
        "doctor": cmd_doctor,
    }
    return int(handlers[args.command](args, settings))


if __name__ == "__main__":
    sys.exit(main())
