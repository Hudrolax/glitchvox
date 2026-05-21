from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import handle_domain_error, handle_unhandled
from app.api.routes_audio import router as audio_router
from app.api.routes_health import router as health_router
from app.api.routes_models import router as models_router
from app.config import load_config
from app.domain.exceptions import GlitchvoxError
from app.infrastructure.registry import TranscriberRegistry
from app.service.transcription_service import TranscriptionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("glitchvox")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    registry = TranscriberRegistry.from_config(cfg)
    await registry.preload(cfg)
    service = TranscriptionService(registry)

    app.state.config = cfg
    app.state.registry = registry
    app.state.service = service

    log.info(
        "glitchvox ready: models=%s default=%s auth=%s",
        registry.list_ids(),
        registry.default_model_id,
        "on" if cfg.api_token else "off",
    )
    yield


app = FastAPI(title="glitchvox", lifespan=lifespan)

app.add_exception_handler(GlitchvoxError, handle_domain_error)
app.add_exception_handler(Exception, handle_unhandled)

app.include_router(health_router)
app.include_router(models_router)
app.include_router(audio_router)
