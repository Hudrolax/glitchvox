from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    registry = request.app.state.registry
    return {
        "status": "ok",
        "models": registry.list_ids(),
        "default_model": registry.default_model_id,
    }
