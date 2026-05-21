from __future__ import annotations

import io
import logging

from app.config import OpenAIConfig
from app.domain.exceptions import UpstreamError
from app.domain.models import Segment, TranscriptionRequest, TranscriptionResult

log = logging.getLogger(__name__)


class OpenAITranscriber:
    """Pass-through to the OpenAI Audio API.

    Uploads the raw bytes the user sent us; no local decoding required.
    """

    needs_decoded_audio = False
    capabilities = {"transcribe", "language_detect"}

    def __init__(self, model_id: str, cfg: OpenAIConfig, api_key: str) -> None:
        self.model_id = model_id
        self._cfg = cfg
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=cfg.base_url)

    async def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        buf = io.BytesIO(req.raw_bytes)
        buf.name = req.filename or "audio.wav"

        # gpt-4o-transcribe currently supports response_format "json" / "text" only.
        # We always ask for "json" upstream and shape our own response downstream.
        try:
            resp = await self._client.audio.transcriptions.create(
                model=self._cfg.model_name,
                file=buf,
                language=req.language,
                prompt=req.prompt,
                temperature=req.temperature,
                response_format="json",
            )
        except Exception as e:
            raise UpstreamError(f"OpenAI API failed: {e}") from e

        text = getattr(resp, "text", "") or ""
        language = getattr(resp, "language", None)
        duration = getattr(resp, "duration", None)
        raw_segments = getattr(resp, "segments", None) or []

        segs: list[Segment] = []
        for s in raw_segments:
            # SDK returns objects with attributes or dicts
            get = (lambda k, d=None: getattr(s, k, d)) if not isinstance(s, dict) else (lambda k, d=None: s.get(k, d))
            segs.append(
                Segment(
                    start=float(get("start", 0.0) or 0.0),
                    end=float(get("end", 0.0) or 0.0),
                    text=str(get("text", "") or ""),
                )
            )

        return TranscriptionResult(
            text=text.strip(),
            language=language,
            duration=float(duration) if duration is not None else None,
            segments=segs,
            model_id=self.model_id,
        )
