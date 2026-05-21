from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .models import TranscriptionRequest, TranscriptionResult

Capability = Literal["transcribe", "segments", "language_detect"]


@runtime_checkable
class Transcriber(Protocol):
    model_id: str
    capabilities: set[Capability]
    needs_decoded_audio: bool  # True for local backends, False for API passthrough

    async def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult: ...

    async def warmup(self) -> None:
        """Load model weights into memory (e.g. GPU VRAM) without running inference.
        For passthrough backends this is a no-op."""
        ...
