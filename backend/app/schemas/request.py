"""
============================================================================
Request Schemas - Pydantic Models for API Requests
============================================================================
Defines the structure and validation for incoming API requests.

Benefits of Pydantic:
    - Automatic validation
    - Type safety
    - Auto-generated API documentation
    - Clear error messages

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Base request model for deepfake prediction.
    
    This is currently unused since we use multipart/form-data for file uploads,
    but included for potential JSON-based requests in the future.
    """
    # Future: Add fields for batch processing, options, etc.
    return_metadata: bool = Field(
        default=True,
        description="Whether to include detailed metadata in response"
    )
    
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom confidence threshold for predictions"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "return_metadata": True,
                "confidence_threshold": 0.5
            }
        }
    }


class ImagePredictionOptions(BaseModel):
    """
    Options for image deepfake detection.
    """
    # Future enhancements
    detect_faces: bool = Field(
        default=True,
        description="Whether to detect and analyze faces in the image"
    )
    
    return_heatmap: bool = Field(
        default=False,
        description="Whether to generate attention heatmap showing suspicious regions"
    )


class VideoPredictionOptions(BaseModel):
    """
    Options for video deepfake detection.
    """
    fps_sample_rate: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Frames per second to sample (overrides default)"
    )
    
    max_frames: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description="Maximum number of frames to analyze"
    )
    
    analyze_audio: bool = Field(
        default=False,
        description="Whether to also analyze audio track for inconsistencies"
    )
    
    @field_validator("fps_sample_rate")
    @classmethod
    def validate_fps(cls, v):
        if v is not None and v > 30:
            raise ValueError("FPS sample rate cannot exceed 30")
        return v


class AudioPredictionOptions(BaseModel):
    """
    Options for audio deepfake detection.
    """
    segment_duration: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Duration of audio segments in seconds"
    )
    
    sample_rate: Optional[int] = Field(
        default=None,
        description="Target sample rate for audio processing"
    )
    
    return_spectrogram: bool = Field(
        default=False,
        description="Whether to return mel-spectrogram visualization"
    )


# =============================================================================
# Batch Processing Requests (Future Enhancement)
# =============================================================================

class BatchPredictionRequest(BaseModel):
    """
    Request model for batch processing multiple files.
    
    Future feature: Process multiple files in a single request.
    """
    file_urls: list[str] = Field(
        description="URLs or paths to files for batch processing"
    )
    
    detection_type: str = Field(
        description="Type of detection: image, video, or audio"
    )
    
    options: Optional[dict] = Field(
        default=None,
        description="Detection options"
    )
    
    @field_validator("detection_type")
    @classmethod
    def validate_detection_type(cls, v):
        allowed = ["image", "video", "audio"]
        if v not in allowed:
            raise ValueError(f"detection_type must be one of {allowed}")
        return v
