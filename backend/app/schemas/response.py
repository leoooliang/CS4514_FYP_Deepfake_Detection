"""
Pydantic response models for the detection API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(description="Real or Deepfake")
    is_deepfake: bool = Field(description="True if Deepfake")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of Deepfake")
    probabilities: Dict[str, float]
    processing_time_seconds: float = Field(description="Total processing time in seconds")
    inference_time_ms: float = Field(description="Model inference time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(description="Additional metadata about the detection")
    record_id: Optional[str] = Field(description="Database record ID (UUID)")


class ImagePredictionResponse(PredictionResponse):
    pass


class VideoPredictionResponse(PredictionResponse):
    pass


class AudioPredictionResponse(PredictionResponse):
    pass


class ErrorDetail(BaseModel):
    loc: Optional[List[str]] = None
    msg: str
    type: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    models: Dict[str, str]
    system: Optional[Dict[str, float]] = None


class BatchPredictionItem(BaseModel):
    file_id: str
    status: str
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[BatchPredictionItem]
    total_processing_time: float
