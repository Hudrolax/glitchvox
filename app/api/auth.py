from __future__ import annotations

from fastapi import Header, Request

from app.domain.exceptions import AuthError


async def bearer_auth(request: Request, authorization: str | None = Header(default=None)) -> None:
    expected: str | None = request.app.state.config.api_token
    if not expected:
        return  # auth disabled
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != expected:
        raise AuthError("Invalid API token")
