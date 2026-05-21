from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.exceptions import GlitchvoxError

log = logging.getLogger(__name__)


def _envelope(message: str, err_type: str, status: int, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": code}},
    )


async def handle_domain_error(_: Request, exc: GlitchvoxError) -> JSONResponse:
    return _envelope(str(exc) or exc.error_type, exc.error_type, exc.http_status)


async def handle_unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return _envelope("Internal server error", "server_error", 500)
