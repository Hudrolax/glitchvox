from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import soundfile as sf

from app.config import GigaAMConfig
from app.domain.exceptions import UpstreamError
from app.domain.models import SAMPLE_RATE, Segment, TranscriptionRequest, TranscriptionResult

log = logging.getLogger(__name__)


class GigaAMTranscriber:
    needs_decoded_audio = True
    capabilities = {"transcribe", "segments"}

    def __init__(self, model_id: str, cfg: GigaAMConfig) -> None:
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
            log.info("Loading GigaAM variant=%s device=%s", self._cfg.variant, self._cfg.device)
            import gigaam

            self._model = await asyncio.to_thread(
                gigaam.load_model, self._cfg.variant, device=self._cfg.device,
            )

    async def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        await self._ensure_loaded()
        assert req.audio is not None, "GigaAM requires decoded audio"
        use_longform = (req.duration_sec or 0.0) > self._cfg.longform_threshold_sec

        def _run() -> TranscriptionResult:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, req.audio, SAMPLE_RATE, subtype="PCM_16")
                tmp.flush()
                path = tmp.name
                try:
                    if use_longform:
                        utterances = self._model.transcribe_longform(path)
                        segs: list[Segment] = []
                        parts: list[str] = []
                        for u in utterances:
                            # gigaam returns dicts: {"transcription": str, "boundaries": (start, end)}
                            text = u.get("transcription") if isinstance(u, dict) else getattr(u, "transcription", "")
                            bounds = (
                                u.get("boundaries")
                                if isinstance(u, dict)
                                else getattr(u, "boundaries", None)
                            )
                            start, end = (bounds or (0.0, 0.0))
                            segs.append(Segment(start=float(start), end=float(end), text=text or ""))
                            if text:
                                parts.append(text)
                        full = " ".join(p.strip() for p in parts).strip()
                        return TranscriptionResult(
                            text=full,
                            language="ru",
                            duration=req.duration_sec,
                            segments=segs,
                            model_id=self.model_id,
                        )
                    else:
                        text = self._model.transcribe(path)
                        text = (text or "").strip()
                        seg = Segment(start=0.0, end=float(req.duration_sec or 0.0), text=text)
                        return TranscriptionResult(
                            text=text,
                            language="ru",
                            duration=req.duration_sec,
                            segments=[seg],
                            model_id=self.model_id,
                        )
                except Exception as e:
                    raise UpstreamError(f"GigaAM failed: {e}") from e

        return await asyncio.to_thread(_run)
