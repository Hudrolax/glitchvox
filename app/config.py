from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class GigaAMConfig(BaseModel):
    backend: Literal["gigaam"]
    variant: Literal["v3_rnnt", "v3_ctc", "v2_rnnt", "v2_ctc"] = "v3_rnnt"
    device: Literal["cuda", "cpu"] = "cuda"
    longform_threshold_sec: float = 30.0


class FasterWhisperConfig(BaseModel):
    backend: Literal["faster_whisper"]
    model_name: str = "large-v3"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5


class OpenAIConfig(BaseModel):
    backend: Literal["openai"]
    model_name: str = "gpt-4o-transcribe"
    base_url: str | None = None


ModelConfig = Annotated[
    GigaAMConfig | FasterWhisperConfig | OpenAIConfig,
    Field(discriminator="backend"),
]


class ServerConfig(BaseModel):
    max_upload_mb: int = 100
    request_timeout_sec: int = 600


class AppConfig(BaseModel):
    default_model: str
    models: dict[str, ModelConfig]
    server: ServerConfig = ServerConfig()

    # Env-derived secrets (not in YAML)
    api_token: str | None = None
    openai_api_key: str | None = None

    @model_validator(mode="after")
    def _check_default(self) -> "AppConfig":
        if self.default_model not in self.models:
            raise ValueError(
                f"default_model={self.default_model!r} is not declared in models"
            )
        return self


def _coerce_raw_models(raw: dict[str, Any]) -> dict[str, Any]:
    """Pydantic discriminated union expects each model dict to include `backend`.
    YAML already has it — this is just a sanity check with a friendlier error.
    """
    out: dict[str, Any] = {}
    for name, cfg in (raw.get("models") or {}).items():
        if not isinstance(cfg, dict) or "backend" not in cfg:
            raise ValueError(f"model {name!r}: missing 'backend' key")
        out[name] = cfg
    raw["models"] = out
    return raw


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or os.environ.get("CONFIG_PATH", "config/config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _coerce_raw_models(raw)

    api_token = os.environ.get("API_TOKEN") or None
    openai_api_key = os.environ.get("OPENAI_API_KEY") or None

    # Drop OpenAI models from config if key is absent — they cannot be served.
    if not openai_api_key:
        raw["models"] = {
            k: v for k, v in raw["models"].items() if v.get("backend") != "openai"
        }
        if raw.get("default_model") not in raw["models"]:
            if not raw["models"]:
                raise ValueError("No usable models: OPENAI_API_KEY is unset and no local models configured")
            raw["default_model"] = next(iter(raw["models"]))

    return AppConfig(
        **raw,
        api_token=api_token,
        openai_api_key=openai_api_key,
    )
