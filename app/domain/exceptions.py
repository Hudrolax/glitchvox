from __future__ import annotations


class GlitchvoxError(Exception):
    """Base error."""

    http_status: int = 500
    error_type: str = "server_error"


class ModelNotAvailable(GlitchvoxError):
    http_status = 404
    error_type = "model_not_found"


class AudioDecodeError(GlitchvoxError):
    http_status = 400
    error_type = "invalid_audio"


class UnsupportedFormat(GlitchvoxError):
    http_status = 400
    error_type = "invalid_request_error"


class UpstreamError(GlitchvoxError):
    http_status = 502
    error_type = "upstream_error"


class AuthError(GlitchvoxError):
    http_status = 401
    error_type = "invalid_api_key"
