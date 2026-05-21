from __future__ import annotations

import logging
import time

from app.config import AppConfig, FasterWhisperConfig, GigaAMConfig, OpenAIConfig
from app.domain.exceptions import ModelNotAvailable
from app.domain.transcriber import Transcriber

from .transcribers.faster_whisper import FasterWhisperTranscriber
from .transcribers.gigaam import GigaAMTranscriber
from .transcribers.openai_api import OpenAITranscriber

log = logging.getLogger(__name__)


class TranscriberRegistry:
    """Holds Transcriber instances. Weights are loaded lazily on first use,
    so process start is cheap even with many models declared.
    """

    def __init__(self, transcribers: dict[str, Transcriber], default_model_id: str) -> None:
        self._transcribers = transcribers
        self.default_model_id = default_model_id

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "TranscriberRegistry":
        instances: dict[str, Transcriber] = {}
        for model_id, mcfg in cfg.models.items():
            if isinstance(mcfg, GigaAMConfig):
                instances[model_id] = GigaAMTranscriber(model_id, mcfg)
            elif isinstance(mcfg, FasterWhisperConfig):
                instances[model_id] = FasterWhisperTranscriber(model_id, mcfg)
            elif isinstance(mcfg, OpenAIConfig):
                if not cfg.openai_api_key:
                    log.warning("Skipping %s: OPENAI_API_KEY is not set", model_id)
                    continue
                instances[model_id] = OpenAITranscriber(model_id, mcfg, cfg.openai_api_key)
            else:
                raise RuntimeError(f"Unknown backend for model {model_id}")

        if not instances:
            raise RuntimeError("No transcribers could be initialized")

        default = cfg.default_model if cfg.default_model in instances else next(iter(instances))
        log.info("Registry: %s, default=%s", list(instances), default)
        return cls(instances, default)

    def has(self, model_id: str | None) -> bool:
        return bool(model_id) and model_id in self._transcribers

    def get(self, model_id: str) -> Transcriber:
        if model_id not in self._transcribers:
            raise ModelNotAvailable(f"Model {model_id!r} is not available")
        return self._transcribers[model_id]

    def resolve(self, requested: str | None) -> Transcriber:
        """OpenAI-style routing: honor requested model if known, else fall back."""
        if self.has(requested):
            return self._transcribers[requested]  # type: ignore[index]
        return self._transcribers[self.default_model_id]

    def list_ids(self) -> list[str]:
        return list(self._transcribers)

    async def preload(self, cfg: AppConfig) -> None:
        """Eagerly load model weights for every model with `preload: true`.

        Done sequentially so two models don't race for the same GPU and
        risk OOM during cuDNN/CTranslate2 allocation. If a model fails
        to preload, it is dropped from the registry — the service stays
        up serving whatever loaded successfully.
        """
        for model_id in list(self._transcribers.keys()):
            mcfg = cfg.models.get(model_id)
            if mcfg is None or not getattr(mcfg, "preload", False):
                continue
            transcriber = self._transcribers[model_id]
            log.info("Preloading %s...", model_id)
            t0 = time.monotonic()
            try:
                await transcriber.warmup()
            except Exception as e:
                log.error(
                    "Failed to preload %s: %s — model disabled for this run",
                    model_id, e,
                )
                self._transcribers.pop(model_id, None)
                continue
            log.info("Preloaded %s in %.1fs", model_id, time.monotonic() - t0)

        # If the default was the one that failed, fall back to the first survivor.
        if self.default_model_id not in self._transcribers:
            if not self._transcribers:
                raise RuntimeError("All preload attempts failed; no usable models")
            new_default = next(iter(self._transcribers))
            log.warning(
                "Default model %r is unavailable; falling back to %r",
                self.default_model_id, new_default,
            )
            self.default_model_id = new_default
