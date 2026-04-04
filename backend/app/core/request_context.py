"""
Per-request context for logging (HTTP correlation ID).

Uses contextvars so values stay isolated across concurrent asyncio tasks.
"""

from contextvars import ContextVar, Token
from typing import Optional
import uuid

_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def bind_request_id(request_id: Optional[str] = None) -> tuple[str, Token]:
    rid = request_id or uuid.uuid4().hex[:8]
    return rid, _request_id.set(rid)


def reset_request_id(token: Token) -> None:
    _request_id.reset(token)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def format_request_id_prefix() -> str:
    rid = _request_id.get()
    if not rid:
        return ""
    return f"[Req: {rid}] "
