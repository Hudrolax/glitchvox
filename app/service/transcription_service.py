from __future__ import annotations

import logging

from app.domain.models import TranscriptionRequest, TranscriptionResult
from app.infrastructure.audio import decode_to_pcm16k
from app.infrastructure.registry import TranscriberRegistry

log = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, registry: TranscriberRegistry) -> None:
        self._registry = registry

    async def transcribe(self, requested_model: str | None, req: TranscriptionRequest) -> TranscriptionResult:
        transcriber = self._registry.resolve(requested_model)

        if requested_model and not self._registry.has(requested_model):
            log.info(
                "Requested model %r not available, falling back to %r",
                requested_model, transcriber.model_id,
            )

        if transcriber.needs_decoded_audio and req.audio is None:
            req.audio, req.duration_sec = await decode_to_pcm16k(req.raw_bytes)

        return await transcriber.transcribe(req)
