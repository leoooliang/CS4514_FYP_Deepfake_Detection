"""
Pydantic models for telemetry endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectionRecordResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    file_name: str
    file_type: str
    file_size: int
    detection_score: float = Field(ge=0.0, le=1.0)
    classification: str
    model_version: str
    timestamp: str
    processing_duration: float
    session_id: Optional[str] = None
    media_path: Optional[str] = None


class DetectionHistoryResponse(BaseModel):
    total: int
    records: List[Dict[str, Any]]


class PlatformStatsResponse(BaseModel):
    total_scans: int
    deepfakes_detected: int
    real_media_detected: int
    avg_processing_duration: float
    scans_by_type: Dict[str, int]
    classification_breakdown: Dict[str, int]
