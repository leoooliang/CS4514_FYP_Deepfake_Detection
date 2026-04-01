"""
Schemas Package - Pydantic Models for Request/Response Validation
"""

from app.schemas.request import (
    PredictionRequest,
    ImagePredictionOptions,
    VideoPredictionOptions,
    AudioPredictionOptions,
    BatchPredictionRequest
)

from app.schemas.response import (
    PredictionResponse,
    ImagePredictionResponse,
    VideoPredictionResponse,
    AudioPredictionResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    BatchPredictionResponse,
    BatchPredictionItem
)

__all__ = [
    # Request models
    "PredictionRequest",
    "ImagePredictionOptions",
    "VideoPredictionOptions",
    "AudioPredictionOptions",
    "BatchPredictionRequest",
    
    # Response models
    "PredictionResponse",
    "ImagePredictionResponse",
    "VideoPredictionResponse",
    "AudioPredictionResponse",
    "ErrorResponse",
    "ErrorDetail",
    "HealthResponse",
    "BatchPredictionResponse",
    "BatchPredictionItem",
]
