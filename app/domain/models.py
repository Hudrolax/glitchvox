from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ResponseFormat = Literal["json", "text", "verbose_json", "srt", "vtt"]

SAMPLE_RATE = 16_000


@dataclass
class TranscriptionRequest:
    """Single transcription job passed from the API layer to a Transcriber.

    Carries both raw upload bytes (needed by passthrough backends like OpenAI)
    and lazily-decoded PCM audio (needed by local backends). Decoding is done
    once in the API layer if any local backend will consume the request.
    """

    raw_bytes: bytes
    filename: str
    content_type: str | None

    audio: np.ndarray | None = None  # mono float32, 16 kHz
    duration_sec: float | None = None

    language: str | None = None
    prompt: str | None = None
    temperature: float = 0.0
    response_format: ResponseFormat = "json"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[Segment] = field(default_factory=list)
    model_id: str = ""
