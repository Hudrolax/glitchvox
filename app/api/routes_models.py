from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.api.auth import bearer_auth
from app.config import OpenAIConfig

router = APIRouter(prefix="/v1", dependencies=[Depends(bearer_auth)])


@router.get("/models")
async def list_models(request: Request) -> dict:
    cfg = request.app.state.config
    registry = request.app.state.registry
    now = int(time.time())
    data = []
    for model_id in registry.list_ids():
        owned_by = "openai" if isinstance(cfg.models[model_id], OpenAIConfig) else "local"
        data.append({
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": owned_by,
        })
    return {"object": "list", "data": data}
