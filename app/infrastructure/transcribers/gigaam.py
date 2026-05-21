from __future__ import annotations

import asyncio
import logging
import tempfile

import soundfile as sf

from app.config import GigaAMConfig
from app.domain.exceptions import UpstreamError
from app.domain.models import SAMPLE_RATE, Segment, TranscriptionRequest, TranscriptionResult

log = logging.getLogger(__name__)

# `.transcribe(...)` upstream cap. Longer audio must go through `.transcribe_longform`.
_SHORT_LIMIT_SEC = 25.0


class GigaAMTranscriber:
    needs_decoded_audio = True
    capabilities = {"transcribe", "segments"}

    def __init__(self, model_id: str, cfg: GigaAMConfig) -> None:
        self.model_id = model_id
        self._cfg = cfg
        self._model = None
        self._lock = asyncio.Lock()

    async def warmup(self) -> None:
        await self._ensure_loaded()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            log.info("Loading GigaAM variant=%s device=%s", self._cfg.variant, self._cfg.device)
            import gigaam

            def _load():
                m = gigaam.load_model(self._cfg.variant)
                # `load_model` does not expose a device arg yet; honor cfg by moving the
                # underlying torch module.
                try:
                    m.to(self._cfg.device)
                except Exception:
                    pass
                return m

            self._model = await asyncio.to_thread(_load)

    async def transcribe(self, req: TranscriptionRequest) -> TranscriptionResult:
        await self._ensure_loaded()
        assert req.audio is not None, "GigaAM requires decoded audio"

        duration = req.duration_sec or 0.0
        # `.transcribe_longform` requires HF_TOKEN + the [longform] extras; we don't
        # ship that in the base image. If we got long audio and longform isn't
        # available, fall back to a single `.transcribe` call (model still accepts
        # the audio, just without VAD-aware segmentation) and surface a warning.
        use_longform = (
            duration > _SHORT_LIMIT_SEC
            and duration > self._cfg.longform_threshold_sec
        )

        def _run() -> TranscriptionResult:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, req.audio, SAMPLE_RATE, subtype="PCM_16")
                tmp.flush()
                path = tmp.name
                try:
                    if use_longform:
                        try:
                            segments_raw = self._model.transcribe_longform(path)
                        except Exception as e:
                            log.warning(
                                "GigaAM longform unavailable (%s); falling back to short transcribe",
                                e,
                            )
                            segments_raw = None
                        if segments_raw is not None:
                            segs: list[Segment] = []
                            parts: list[str] = []
                            for s in segments_raw:
                                start = float(getattr(s, "start", 0.0) or 0.0)
                                end = float(getattr(s, "end", 0.0) or 0.0)
                                text = getattr(s, "text", None) or ""
                                segs.append(Segment(start=start, end=end, text=text))
                                if text:
                                    parts.append(text)
                            full = " ".join(p.strip() for p in parts).strip()
                            return TranscriptionResult(
                                text=full,
                                language="ru",
                                duration=duration,
                                segments=segs,
                                model_id=self.model_id,
                            )

                    raw = self._model.transcribe(path)
                    # `.transcribe` returns either a plain string or an object with .text
                    text = getattr(raw, "text", None) if not isinstance(raw, str) else raw
                    if text is None:
                        text = str(raw)
                    text = text.strip()
                    seg = Segment(start=0.0, end=duration, text=text)
                    return TranscriptionResult(
                        text=text,
                        language="ru",
                        duration=duration,
                        segments=[seg],
                        model_id=self.model_id,
                    )
                except Exception as e:
                    raise UpstreamError(f"GigaAM failed: {e}") from e

        return await asyncio.to_thread(_run)
