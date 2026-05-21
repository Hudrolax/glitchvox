from __future__ import annotations

import asyncio
import logging

from app.config import FasterWhisperConfig
from app.domain.exceptions import UpstreamError
from app.domain.models import Segment, TranscriptionRequest, TranscriptionResult

log = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    needs_decoded_audio = True
    capabilities = {"transcribe", "segments", "language_detect"}

    def __init__(self, model_id: str, cfg: FasterWhisperConfig) -> None:
        self.model_id = model_id
        self._cfg = cfg
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            log.info("Loading faster-whisper model_name=%s device=%s", self._cfg.model_name, self._cfg.device)
            from faster_whisper import WhisperModel

            self._model = await asyncio.to_thread(
                WhisperModel,
                self._cfg.model_name,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
            )

    async def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        await self._ensure_loaded()
        assert req.audio is not None, "FasterWhisper requires decoded audio"

        def _run() -> TranscriptionResult:
            try:
                segments_iter, info = self._model.transcribe(
                    req.audio,
                    language=req.language,
                    initial_prompt=req.prompt,
                    temperature=req.temperature,
                    beam_size=self._cfg.beam_size,
                    vad_filter=True,
                )
            except Exception as e:
                raise UpstreamError(f"faster-whisper failed: {e}") from e

            segs: list[Segment] = []
            chunks: list[str] = []
            for s in segments_iter:
                segs.append(Segment(start=float(s.start), end=float(s.end), text=s.text))
                chunks.append(s.text)
            text = "".join(chunks).strip()
            return TranscriptionResult(
                text=text,
                language=info.language,
                duration=float(info.duration),
                segments=segs,
                model_id=self.model_id,
            )

        return await asyncio.to_thread(_run)
