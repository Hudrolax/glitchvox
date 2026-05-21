from __future__ import annotations

from typing import get_args

from app.domain.models import ResponseFormat

ALLOWED_FORMATS: tuple[str, ...] = get_args(ResponseFormat)
