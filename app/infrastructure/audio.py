from __future__ import annotations

import asyncio
import subprocess

import numpy as np

from app.domain.exceptions import AudioDecodeError
from app.domain.models import SAMPLE_RATE


def _decode_sync(raw: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 PCM at 16 kHz via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, input=raw, capture_output=True, check=True,
        )
    except FileNotFoundError as e:
        raise AudioDecodeError("ffmpeg binary not found in container") from e
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed"
        raise AudioDecodeError(f"Cannot decode audio: {msg}") from e

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise AudioDecodeError("Decoded audio is empty")
    return pcm.astype(np.float32) / 32768.0


async def decode_to_pcm16k(raw: bytes) -> tuple[np.ndarray, float]:
    """Async wrapper around the blocking ffmpeg decode."""
    audio = await asyncio.to_thread(_decode_sync, raw)
    duration = float(audio.shape[0]) / SAMPLE_RATE
    return audio, duration
