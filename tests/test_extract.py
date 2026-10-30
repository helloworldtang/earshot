"""extract 模块单测:ffmpeg 真进程(无 ffmpeg 环境跳过)。"""

import subprocess
import wave
from pathlib import Path

import pytest

from earshot.extract import ffmpeg_available, to_wav16k, wav_duration

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="本机无 ffmpeg")


def _make_sine(path: Path, seconds: float = 1.0, rate: int = 44100, channels: int = 2) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_to_wav16k_downmixes_and_resamples(tmp_path: Path) -> None:
    src = tmp_path / "in.wav"
    _make_sine(src, seconds=1.0, rate=44100, channels=2)
    dst = tmp_path / "out16k.wav"
    to_wav16k(src, dst)
    with wave.open(str(dst), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1


def test_to_wav16k_accepts_video_container(tmp_path: Path) -> None:
    src = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:a",
            "aac",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    dst = tmp_path / "out16k.wav"
    to_wav16k(src, dst)
    assert wav_duration(dst) == pytest.approx(2.0, abs=0.1)


def test_to_wav16k_error_on_garbage(tmp_path: Path) -> None:
    src = tmp_path / "garbage.wav"
    src.write_bytes(b"not a wav at all")
    from earshot.extract import ExtractError

    with pytest.raises(ExtractError):
        to_wav16k(src, tmp_path / "out.wav")
