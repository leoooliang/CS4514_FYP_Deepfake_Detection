"""
Centralised file validation and security helpers.

Helper functions:
- validate_file_size
- validate_file_extension
- validate_mime_type
- validate_image_integrity
- validate_image_file
- validate_video_file
- validate_audio_file
- sanitize_filename
- validate_upload_safety
"""

from pathlib import Path
from typing import Tuple

from loguru import logger
from PIL import Image

from app.config import settings

try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False


def validate_file_size(file_size: int, max_size: int, file_type: str = "file") -> None:
    if file_size > max_size:
        raise ValueError(
            f"{file_type.capitalize()} size ({file_size / (1024 * 1024):.2f} MB) "
            f"exceeds maximum ({max_size / (1024 * 1024):.2f} MB)"
        )


def validate_file_extension(filename: str, allowed: list) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"Extension '{ext}' not allowed. Allowed: {', '.join(allowed)}")


def detect_mime_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    _mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".bmp": "image/bmp", ".webp": "image/webp",
        ".mp4": "video/mp4", ".avi": "video/avi", ".mov": "video/quicktime",
        ".mkv": "video/x-matroska", ".webm": "video/webm",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4",
    }
    if HAS_MAGIC:
        try:
            return magic.Magic(mime=True).from_file(file_path)
        except Exception:
            pass
    return _mime_map.get(ext, "application/octet-stream")


def validate_mime_type(file_path: str, allowed_prefixes: list, category: str = "file") -> Tuple[bool, str]:
    mime = detect_mime_type(file_path)
    if any(mime.startswith(p) for p in allowed_prefixes):
        return True, mime
    raise ValueError(f"Invalid {category} type. Expected {', '.join(allowed_prefixes)} but got {mime}")


def validate_image_integrity(file_path: str) -> None:
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {e}")


def validate_image_file(file_path: str, file_size: int, filename: str) -> str:
    validate_file_extension(filename, settings.ALLOWED_IMAGE_EXTENSIONS)
    validate_file_size(file_size, settings.max_image_size_bytes, "image")
    _, mime = validate_mime_type(file_path, ["image/"], "image")
    validate_image_integrity(file_path)
    logger.info("[VALIDATION] Image accepted: file='{}', mime={}, size={:.2f} MB",
                filename, mime, file_size / (1024 * 1024))
    return mime


def validate_video_file(file_path: str, file_size: int, filename: str) -> str:
    validate_file_extension(filename, settings.ALLOWED_VIDEO_EXTENSIONS)
    validate_file_size(file_size, settings.max_video_size_bytes, "video")
    _, mime = validate_mime_type(file_path, ["video/"], "video")
    logger.info("[VALIDATION] Video accepted: file='{}', mime={}, size={:.2f} MB",
                filename, mime, file_size / (1024 * 1024))
    return mime


def validate_audio_file(file_path: str, file_size: int, filename: str) -> str:
    validate_file_extension(filename, settings.ALLOWED_AUDIO_EXTENSIONS)
    validate_file_size(file_size, settings.max_audio_size_bytes, "audio")
    _, mime = validate_mime_type(file_path, ["audio/"], "audio")
    logger.info("[VALIDATION] Audio accepted: file='{}', mime={}, size={:.2f} MB",
                filename, mime, file_size / (1024 * 1024))
    return mime


def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    for ch in ("..", "/", "\\", "<", ">", ":", '"', "|", "?", "*"):
        filename = filename.replace(ch, "_")
    return filename


def validate_upload_safety(filename: str) -> None:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename: path traversal detected")
    if len(filename) > 255:
        raise ValueError("Filename too long (max 255 characters)")
    if "\x00" in filename:
        raise ValueError("Invalid filename: null byte detected")
