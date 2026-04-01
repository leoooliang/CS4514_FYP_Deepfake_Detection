"""
============================================================================
Response Schemas - Pydantic Models for API Responses
============================================================================
Defines the structure of API responses for consistency and documentation.

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class PredictionResponse(BaseModel):
    """
    Base response model for deepfake predictions.
    
    This structure is consistent across all detection types (image/video/audio).
    
    Standardized Field Names:
        - confidence: Confidence score (0-1) - SINGLE SOURCE OF TRUTH
        - processing_time_seconds: Processing duration in seconds
        - inference_time_ms: Inference duration in milliseconds
    """
    prediction: str = Field(
        description="Prediction result: 'real' or 'deepfake'"
    )
    
    is_fake: bool = Field(
        description="Boolean flag: True if deepfake, False if real"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of being a deepfake (0-1). Values >0.5 indicate deepfake, <=0.5 indicate real."
    )
    
    probabilities: Dict[str, float] = Field(
        description="Probability distribution over classes"
    )
    
    processing_time_seconds: float = Field(
        description="Total processing time in seconds"
    )
    
    inference_time_ms: float = Field(
        description="Model inference time in milliseconds"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the detection"
    )
    
    record_id: Optional[str] = Field(
        default=None,
        description="Database record ID (UUID) for this detection"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "deepfake",
                "is_fake": True,
                "confidence": 0.92,
                "probabilities": {
                    "real": 0.08,
                    "deepfake": 0.92
                },
                "processing_time_seconds": 1.23,
                "inference_time_ms": 1230.0,
                "metadata": {
                    "model_type": "ImageDeepfakeNet",
                    "device": "cuda"
                },
                "record_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    }


class ImagePredictionResponse(PredictionResponse):
    """Response model for image detection (same shape as base prediction)."""


class VideoPredictionResponse(PredictionResponse):
    """Response model for video detection (same shape as base prediction)."""


class AudioPredictionResponse(PredictionResponse):
    """Response model for audio detection (same shape as base prediction)."""


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """
    Detailed error information.
    """
    loc: Optional[List[str]] = Field(
        default=None,
        description="Location of the error (for validation errors)"
    )
    
    msg: str = Field(
        description="Error message"
    )
    
    type: str = Field(
        description="Error type identifier"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(
        description="Error category or title"
    )
    
    message: str = Field(
        description="Human-readable error message"
    )
    
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information (for validation errors)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the error"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file format",
                "details": [
                    {
                        "loc": ["file"],
                        "msg": "File must be an image (jpg, png, etc.)",
                        "type": "value_error"
                    }
                ],
                "timestamp": "2026-01-28T10:30:00Z"
            }
        }
    }


# =============================================================================
# Health Check Response
# =============================================================================

class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(
        description="Overall system status: 'healthy' or 'unhealthy'"
    )
    
    timestamp: float = Field(
        description="Unix timestamp of the health check"
    )
    
    models: Dict[str, str] = Field(
        description="Status of each ML model"
    )
    
    system: Optional[Dict[str, float]] = Field(
        default=None,
        description="System resource usage"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": 1706437800.0,
                "models": {
                    "image_detector": "loaded",
                    "video_detector": "loaded",
                    "audio_detector": "loaded"
                },
                "system": {
                    "cpu_percent": 35.2,
                    "memory_percent": 58.7
                }
            }
        }
    }


# =============================================================================
# Batch Processing Response (Future)
# =============================================================================

class BatchPredictionItem(BaseModel):
    """
    Single item in a batch prediction response.
    """
    file_id: str = Field(
        description="Identifier for the processed file"
    )
    
    status: str = Field(
        description="Processing status: 'success' or 'error'"
    )
    
    result: Optional[PredictionResponse] = Field(
        default=None,
        description="Prediction result (if successful)"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch processing.
    """
    total_files: int = Field(
        description="Total number of files processed"
    )
    
    successful: int = Field(
        description="Number of successfully processed files"
    )
    
    failed: int = Field(
        description="Number of failed files"
    )
    
    results: List[BatchPredictionItem] = Field(
        description="Individual results for each file"
    )
    
    total_processing_time: float = Field(
        description="Total processing time in seconds"
    )
