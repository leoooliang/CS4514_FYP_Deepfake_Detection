"""
General-purpose utility helpers:

Helper functions:
- get_file_hash
- format_bytes
"""

import hashlib


def get_file_hash(file_path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()


def format_bytes(value: int) -> str:
    """Format a byte count as a human-readable string (e.g. '1.50 MB')."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"
