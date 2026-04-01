"""
============================================================================
Configuration Management - Environment Variables & Settings
============================================================================
Centralized configuration using Pydantic Settings for type-safe config.

Usage:
    from app.config import settings
    print(settings.API_V1_PREFIX)
============================================================================
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings can be overridden by creating a .env file in the backend directory.
    """
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    APP_NAME: str = "Multimodal Deepfake Detection System"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    VERSION: str = "1.0.0"
    
    # =========================================================================
    # Server Configuration
    # =========================================================================
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    
    # API prefix for versioning
    API_V1_PREFIX: str = "/api/v1"
    
    # =========================================================================
    # CORS Settings
    # =========================================================================
    ALLOWED_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",  # Alternative React port
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000"
        ]
    )
    
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"]
    )
    
    # =========================================================================
    # File Upload Settings
    # =========================================================================
    # Maximum file sizes in MB
    MAX_IMAGE_SIZE_MB: int = Field(default=100)
    MAX_VIDEO_SIZE_MB: int = Field(default=100)
    MAX_AUDIO_SIZE_MB: int = Field(default=100)
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    
    # Temporary upload directory
    UPLOAD_DIR: str = Field(default="./uploads")
    
    # =========================================================================
    # ML Model Paths
    # =========================================================================
    # Paths to the trained model files
    # Models are loaded from backend/models/ directory
    MODEL_DIR: str = Field(default="models")
    
    # Paths to specific trained models
    # These point to the actual trained models from your FYP training
    IMAGE_MODEL_PATH: Optional[str] = Field(
        default="models/best_noise_efficientnet.pth"
    )
    VIDEO_MODEL_PATH: Optional[str] = Field(
        default="models/best_video_tristream.pth"
    )
    AUDIO_MODEL_PATH: Optional[str] = Field(
        default="models/best_audio_cnn_gru.pth"
    )
    
    # =========================================================================
    # Model Inference Settings
    # =========================================================================
    # Device for PyTorch (cpu, cuda, mps)
    # Auto-detect CUDA availability by default
    DEVICE: str = Field(default="auto")
    
    # Batch size for processing
    BATCH_SIZE: int = Field(default=1)
    
    # Number of worker threads
    NUM_WORKERS: int = Field(default=2)
    
    # Confidence threshold for predictions
    CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    
    # =========================================================================
    # Video Processing Settings
    # =========================================================================
    # Extract N frames per second from video
    VIDEO_FPS_SAMPLE_RATE: int = Field(default=1)
    
    # Maximum number of frames to process
    MAX_FRAMES_TO_PROCESS: int = Field(default=100)
    
    # =========================================================================
    # Audio Processing Settings
    # =========================================================================
    # Sample rate for audio processing
    AUDIO_SAMPLE_RATE: int = Field(default=16000)
    
    # Segment duration in seconds (model trained with 4s segments)
    AUDIO_SEGMENT_DURATION: int = Field(default=4)
    
    # =========================================================================
    # Security Settings
    # =========================================================================
    # Secret key for JWT tokens (if implementing authentication)
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production"
    )
    
    # Rate limiting (requests per minute)
    RATE_LIMIT: int = Field(default=30)
    
    # =========================================================================
    # Logging Settings
    # =========================================================================
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")
    
    # =========================================================================
    # Database Settings (for future expansion)
    # =========================================================================
    DATABASE_URL: Optional[str] = Field(default=None)
    
    # =========================================================================
    # Redis Cache Settings (for future expansion)
    # =========================================================================
    REDIS_URL: Optional[str] = Field(default=None)
    CACHE_TTL: int = Field(default=3600)  # Cache time-to-live in seconds
    
    # =========================================================================
    # Validators
    # =========================================================================
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("UPLOAD_DIR", "MODEL_DIR")
    @classmethod
    def create_directories(cls, v):
        """Ensure required directories exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @property
    def max_image_size_bytes(self) -> int:
        """Convert MB to bytes for image size validation."""
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    @property
    def max_video_size_bytes(self) -> int:
        """Convert MB to bytes for video size validation."""
        return self.MAX_VIDEO_SIZE_MB * 1024 * 1024
    
    @property
    def max_audio_size_bytes(self) -> int:
        """Convert MB to bytes for audio size validation."""
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024
    
    # =========================================================================
    # Pydantic Config
    # =========================================================================
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True
    }


# =============================================================================
# Global Settings Instance
# =============================================================================
# This is the single source of truth for all configuration in the application
settings = Settings()


# =============================================================================
# Logging Configuration (Initialize BEFORE device detection)
# =============================================================================
from loguru import logger
import sys

# Configure loguru logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL
)

# Add file logging if specified
if settings.LOG_FILE:
    log_dir = os.path.dirname(settings.LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logger.add(
        settings.LOG_FILE,
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        level=settings.LOG_LEVEL
    )


# =============================================================================
# CUDA/GPU Auto-Detection
# =============================================================================
def get_device() -> str:
    """
    Automatically detect the best available device for PyTorch.
    
    Priority:
        1. CUDA (NVIDIA GPU) if available
        2. MPS (Apple Silicon) if available
        3. CPU as fallback
    
    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    import torch
    
    if settings.DEVICE.lower() == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ CUDA GPU detected: {gpu_name} ({gpu_count} GPU(s) available)")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  PyTorch Version: {torch.__version__}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("✓ Apple MPS (Metal Performance Shaders) detected")
        else:
            device = "cpu"
            logger.warning("⚠ No GPU detected, using CPU (this will be slower)")
        
        # Update settings with detected device
        settings.DEVICE = device
        return device
    else:
        # User specified a device, validate it
        device = settings.DEVICE.lower()
        import torch
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠ CUDA requested but not available, falling back to CPU")
            settings.DEVICE = "cpu"
            return "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("⚠ MPS requested but not available, falling back to CPU")
            settings.DEVICE = "cpu"
            return "cpu"
        
        return device

# Auto-detect device on import
DEVICE = get_device()
