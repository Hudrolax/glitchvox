from __future__ import annotations

import logging

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
