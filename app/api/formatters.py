from __future__ import annotations

from app.domain.models import ResponseFormat, Segment, TranscriptionResult


def _ts_srt(sec: float) -> str:
    if sec < 0:
        sec = 0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(sec: float) -> str:
    return _ts_srt(sec).replace(",", ".")


def _to_srt(segments: list[Segment]) -> str:
    out = []
    for i, s in enumerate(segments, start=1):
        out.append(f"{i}\n{_ts_srt(s.start)} --> {_ts_srt(s.end)}\n{s.text.strip()}\n")
    return "\n".join(out).strip() + "\n"


def _to_vtt(segments: list[Segment]) -> str:
    out = ["WEBVTT", ""]
    for s in segments:
        out.append(f"{_ts_vtt(s.start)} --> {_ts_vtt(s.end)}")
        out.append(s.text.strip())
        out.append("")
    return "\n".join(out).strip() + "\n"


def _verbose_json(r: TranscriptionResult) -> dict:
    return {
        "task": "transcribe",
        "language": r.language,
        "duration": r.duration,
        "text": r.text,
        "segments": [
            {
                "id": i,
                "start": s.start,
                "end": s.end,
                "text": s.text,
            }
            for i, s in enumerate(r.segments)
        ],
    }


def render(fmt: ResponseFormat, r: TranscriptionResult) -> tuple[str | dict, str]:
    """Returns (body, media_type)."""
    if fmt == "text":
        return r.text + ("\n" if not r.text.endswith("\n") else ""), "text/plain; charset=utf-8"
    if fmt == "srt":
        return _to_srt(r.segments), "application/x-subrip; charset=utf-8"
    if fmt == "vtt":
        return _to_vtt(r.segments), "text/vtt; charset=utf-8"
    if fmt == "verbose_json":
        return _verbose_json(r), "application/json"
    # default "json"
    return {"text": r.text}, "application/json"
