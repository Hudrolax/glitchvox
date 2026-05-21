from __future__ import annotations

import logging
from typing import get_args

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from app.api.auth import bearer_auth
from app.api.formatters import render
from app.domain.exceptions import UnsupportedFormat
from app.domain.models import ResponseFormat, TranscriptionRequest

from .schemas import ALLOWED_FORMATS  # noqa: F401  (kept for docs)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Depends(bearer_auth)])

_ALLOWED: tuple[str, ...] = get_args(ResponseFormat)


@router.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> Response:
    if response_format not in _ALLOWED:
        raise UnsupportedFormat(
            f"response_format must be one of {list(_ALLOWED)}, got {response_format!r}"
        )

    cfg = request.app.state.config
    max_bytes = cfg.server.max_upload_mb * 1024 * 1024
    raw = await file.read()
    if len(raw) > max_bytes:
        raise UnsupportedFormat(
            f"File too large: {len(raw)} bytes, max {max_bytes}"
        )
    if not raw:
        raise UnsupportedFormat("Uploaded file is empty")

    req = TranscriptionRequest(
        raw_bytes=raw,
        filename=file.filename or "audio",
        content_type=file.content_type,
        language=language,
        prompt=prompt,
        temperature=temperature,
        response_format=response_format,  # type: ignore[arg-type]
    )

    service = request.app.state.service
    result = await service.transcribe(model, req)

    body, media_type = render(response_format, result)  # type: ignore[arg-type]
    if isinstance(body, dict):
        return JSONResponse(content=body, media_type=media_type)
    if media_type.startswith("text/plain"):
        return PlainTextResponse(content=body, media_type=media_type)
    return Response(content=body, media_type=media_type)
