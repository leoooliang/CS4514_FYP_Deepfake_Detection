"""
============================================================================
Input Validation Utilities
============================================================================
Centralized validation functions for API inputs.

This module provides:
    - File validation (size, type, integrity)
    - MIME type checking
    - Content validation
    - Security checks

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

from typing import Tuple, Optional
from pathlib import Path
from PIL import Image
from loguru import logger

from app.config import settings

# Optional: python-magic for better MIME detection
# Install with: pip install python-magic
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logger.warning("python-magic not installed. Using basic MIME detection.")


# =============================================================================
# File Validation Functions
# =============================================================================

def validate_file_size(file_size: int, max_size: int, file_type: str = "file") -> None:
    """
    Validate file size against maximum allowed.
    
    Args:
        file_size: Size of file in bytes
        max_size: Maximum allowed size in bytes
        file_type: Type of file for error messages
    
    Raises:
        ValueError: If file size exceeds maximum
    """
    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"{file_type.capitalize()} size ({actual_mb:.2f} MB) exceeds "
            f"maximum allowed size ({max_mb:.2f} MB)"
        )


def validate_file_extension(filename: str, allowed_extensions: list) -> None:
    """
    Validate file extension against allowed list.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (with dots)
    
    Raises:
        ValueError: If extension is not allowed
    """
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(
            f"File extension '{ext}' not allowed. "
            f"Allowed extensions: {', '.join(allowed_extensions)}"
        )


def detect_mime_type(file_path: str) -> str:
    """
    Detect actual MIME type of file using magic numbers.
    
    This provides more reliable type detection than just extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        str: MIME type (e.g., 'image/jpeg')
    """
    # Extension-based MIME mapping (fallback)
    ext = Path(file_path).suffix.lower()
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.avi': 'video/avi',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4'
    }
    
    # Try magic-based detection if available
    if HAS_MAGIC:
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(file_path)
        except Exception as e:
            logger.warning(f"Magic MIME detection failed: {str(e)}, using extension-based")
    
    return mime_map.get(ext, 'application/octet-stream')


def validate_mime_type(
    file_path: str,
    allowed_types: list,
    category: str = "file"
) -> Tuple[bool, str]:
    """
    Validate file MIME type against allowed types.
    
    Args:
        file_path: Path to file
        allowed_types: List of allowed MIME type prefixes (e.g., ['image/', 'video/'])
        category: File category for error messages
    
    Returns:
        Tuple[bool, str]: (is_valid, mime_type)
    
    Raises:
        ValueError: If MIME type is not allowed
    """
    mime_type = detect_mime_type(file_path)
    
    for allowed in allowed_types:
        if mime_type.startswith(allowed):
            return True, mime_type
    
    raise ValueError(
        f"Invalid {category} type. Expected {', '.join(allowed_types)} "
        f"but got {mime_type}"
    )


def validate_image_integrity(file_path: str) -> None:
    """
    Validate image file integrity by attempting to open it.
    
    Args:
        file_path: Path to image file
    
    Raises:
        ValueError: If image is corrupted or invalid
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify it's a valid image
        # Reopen after verify (verify closes the file)
        with Image.open(file_path) as img:
            img.load()  # Actually load the image data
        logger.debug(f"Image integrity validated: {file_path}")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {str(e)}")


# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_image_file(file_path: str, file_size: int, filename: str) -> str:
    """
    Comprehensive validation for image files.
    
    Args:
        file_path: Path to uploaded file
        file_size: Size of file in bytes
        filename: Original filename
    
    Returns:
        str: Detected MIME type
    
    Raises:
        ValueError: If validation fails
    """
    # Validate extension
    validate_file_extension(filename, settings.ALLOWED_IMAGE_EXTENSIONS)
    
    # Validate size
    validate_file_size(file_size, settings.max_image_size_bytes, "image")
    
    # Validate MIME type
    is_valid, mime_type = validate_mime_type(
        file_path,
        ['image/'],
        "image"
    )
    
    # Validate image integrity
    validate_image_integrity(file_path)
    
    logger.info(f"Image file validated: {filename} ({mime_type})")
    return mime_type


def validate_video_file(file_path: str, file_size: int, filename: str) -> str:
    """
    Comprehensive validation for video files.
    
    Args:
        file_path: Path to uploaded file
        file_size: Size of file in bytes
        filename: Original filename
    
    Returns:
        str: Detected MIME type
    
    Raises:
        ValueError: If validation fails
    """
    # Validate extension
    validate_file_extension(filename, settings.ALLOWED_VIDEO_EXTENSIONS)
    
    # Validate size
    validate_file_size(file_size, settings.max_video_size_bytes, "video")
    
    # Validate MIME type
    is_valid, mime_type = validate_mime_type(
        file_path,
        ['video/'],
        "video"
    )
    
    logger.info(f"Video file validated: {filename} ({mime_type})")
    return mime_type


def validate_audio_file(file_path: str, file_size: int, filename: str) -> str:
    """
    Comprehensive validation for audio files.
    
    Args:
        file_path: Path to uploaded file
        file_size: Size of file in bytes
        filename: Original filename
    
    Returns:
        str: Detected MIME type
    
    Raises:
        ValueError: If validation fails
    """
    # Validate extension
    validate_file_extension(filename, settings.ALLOWED_AUDIO_EXTENSIONS)
    
    # Validate size
    validate_file_size(file_size, settings.max_audio_size_bytes, "audio")
    
    # Validate MIME type
    is_valid, mime_type = validate_mime_type(
        file_path,
        ['audio/'],
        "audio"
    )
    
    logger.info(f"Audio file validated: {filename} ({mime_type})")
    return mime_type


# =============================================================================
# Security Validation
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
    
    Returns:
        str: Sanitized filename
    """
    # Remove path components
    filename = Path(filename).name
    
    # Remove potentially dangerous characters
    dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    return filename


def validate_upload_safety(filename: str) -> None:
    """
    Validate upload for security concerns.
    
    Args:
        filename: Filename to validate
    
    Raises:
        ValueError: If filename contains suspicious patterns
    """
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    # Check for extremely long filenames
    if len(filename) > 255:
        raise ValueError("Filename too long (max 255 characters)")
    
    # Check for null bytes
    if '\x00' in filename:
        raise ValueError("Invalid filename: null byte detected")

