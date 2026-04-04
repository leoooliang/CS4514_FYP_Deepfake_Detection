"""
Centralized configuration.

Usage:
    from app.config import settings, DEVICE
"""

import inspect
import logging
import os
import sys
from typing import Any, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from loguru import logger

from app.core.access_log import effective_uvicorn_access_level
from app.core.request_context import format_request_id_prefix


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    """

    # Application
    APP_NAME: str = "Deepfake Detection System"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    VERSION: str = "1.0.0"

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    API_V1_PREFIX: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
    )
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"])

    # File uploads
    MAX_IMAGE_SIZE_MB: int = Field(default=100)
    MAX_VIDEO_SIZE_MB: int = Field(default=100)
    MAX_AUDIO_SIZE_MB: int = Field(default=100)

    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]

    UPLOAD_DIR: str = Field(default="./uploads")

    # ML model paths
    MODEL_DIR: str = Field(default="models")
    IMAGE_SPATIAL_MODEL_PATH: Optional[str] = Field(default="models/best_clip.pth")
    IMAGE_NOISE_MODEL_PATH: Optional[str] = Field(default="models/best_noise_efficientnet.pth")
    VIDEO_MODEL_PATH: Optional[str] = Field(default="models/best_video_tristream.pth")
    AUDIO_MODEL_PATH: Optional[str] = Field(default="models/best_audio_cnn_gru.pth")

    # Inference
    DEVICE: str = Field(default="auto")
    BATCH_SIZE: int = Field(default=1)
    NUM_WORKERS: int = Field(default=2)
    CONFIDENCE_THRESHOLD: float = Field(default=0.5)

    # Video processing
    VIDEO_FPS_SAMPLE_RATE: int = Field(default=1)
    MAX_FRAMES_TO_PROCESS: int = Field(default=100)

    # Audio processing
    AUDIO_SAMPLE_RATE: int = Field(default=16000)
    AUDIO_SEGMENT_DURATION: int = Field(default=4)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./sql_app.db")

    # Validators
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("UPLOAD_DIR", "MODEL_DIR")
    @classmethod
    def create_directories(cls, v):
        os.makedirs(v, exist_ok=True)
        return v

    # Derived properties
    @property
    def max_image_size_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def max_video_size_bytes(self) -> int:
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


settings = Settings()

logger.remove()


def _loguru_request_patcher(record: Any) -> None:
    record["extra"]["request_id"] = format_request_id_prefix()


_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{extra[request_id]}{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{extra[request_id]}{message}"
)

logger.configure(extra={"request_id": ""}, patcher=_loguru_request_patcher)

logger.add(
    sys.stderr,
    format=_CONSOLE_FORMAT,
    level=settings.LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=settings.DEBUG,
)

if settings.LOG_FILE:
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logger.add(
        settings.LOG_FILE,
        format=_FILE_FORMAT,
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        level=settings.LOG_LEVEL,
        backtrace=True,
        diagnose=False,
    )


def _attach_uvicorn_logs_to_loguru() -> None:
    """
    Send uvicorn's logging through loguru so lines include the same date/time as app logs.
    """

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if record.name == "uvicorn.access":
                level_name = effective_uvicorn_access_level(msg, record.levelname)
            else:
                level_name = record.levelname
            try:
                level = logger.level(level_name).name
            except ValueError:
                level = str(record.levelno)
            frame, depth = inspect.currentframe(), 2
            while frame is not None and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, msg)

    h = InterceptHandler()
    level_map = {
        "TRACE": logging.DEBUG,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    py_level = level_map.get(settings.LOG_LEVEL.upper(), logging.INFO)
    for name in ("uvicorn", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.addHandler(h)
        lg.propagate = False
        lg.setLevel(py_level)
    logging.getLogger("uvicorn.error").setLevel(py_level)


_attach_uvicorn_logs_to_loguru()


def _detect_device() -> str:
    """
    Return the best available PyTorch device string.
    """
    
    import torch

    requested = settings.DEVICE.lower()

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        logger.info("Device explicitly set to '{}'", requested)
        return requested

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info("CUDA GPU detected: {} ({:.0f} MB VRAM)", gpu_name, vram_mb)
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Apple MPS backend detected")
        return "mps"

    logger.warning("No GPU detected, using CPU")
    return "cpu"


DEVICE: str = _detect_device()
