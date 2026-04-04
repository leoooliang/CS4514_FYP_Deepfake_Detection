"""
Helpers for uvicorn access log filtering (high-frequency polling / static files).
"""

import re

# e.g. 127.0.0.1:12345 - "GET /api/v1/telemetry/history HTTP/1.1" 200 OK
_ACCESS_PATH_RE = re.compile(r'"[A-Z]+\s+([^\s?"]+)')
_ACCESS_STATUS_RE = re.compile(r"HTTP/[\d.]+\s*\"\s+(\d{3})")


def uvicorn_access_should_log_at_debug(message: str) -> bool:
    """
    True when this access line should be emitted at DEBUG instead of INFO (successful telemetry or static media fetches).
    """
    
    sm = _ACCESS_STATUS_RE.search(message)
    if not sm or sm.group(1) not in ("200", "304"):
        return False
    pm = _ACCESS_PATH_RE.search(message)
    if not pm:
        return False
    path: str = pm.group(1)
    if path.startswith("/media"):
        return True
    if "/telemetry/" in path:
        return True
    return False


def effective_uvicorn_access_level(message: str, default_level: str) -> str:
    if uvicorn_access_should_log_at_debug(message):
        return "DEBUG"
    return default_level
