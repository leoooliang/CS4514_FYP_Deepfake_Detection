"""
============================================================================
Telemetry Schemas - Response Models for History and Statistics
============================================================================
Pydantic models for telemetry endpoints.

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DetectionRecordResponse(BaseModel):
    """
    Response model for a single detection record.
    
    Represents the stored metadata and results from a detection scan.
    
    Note: This schema matches the database model's to_dict() output.
    """
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "file_name": "sample_image.jpg",
                "file_type": "image",
                "file_size": 2048576,
                "detection_score": 0.92,
                "classification": "Fake",
                "model_version": "ImageDeepfakeNet-v1.0",
                "timestamp": "2026-03-24T10:30:00Z",
                "processing_duration": 1.234,
                "session_id": "session_abc123",
                "media_path": "/media/abc12345_sample_image.jpg"
            }
        }
    }
    
    id: str = Field(
        description="Unique identifier (UUID) for the record"
    )
    
    file_name: str = Field(
        description="Original filename (sanitized)"
    )
    
    file_type: str = Field(
        description="Media type: 'image', 'audio', or 'video'"
    )
    
    file_size: int = Field(
        description="File size in bytes"
    )
    
    detection_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    
    classification: str = Field(
        description="Classification result: 'Real' or 'Fake'"
    )
    
    model_version: str = Field(
        description="Version or identifier of the ML model used"
    )
    
    timestamp: str = Field(
        description="ISO 8601 timestamp when detection was performed"
    )
    
    processing_duration: float = Field(
        description="Processing time in seconds"
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for anonymous user tracking"
    )
    
    media_path: Optional[str] = Field(
        default=None,
        description="URL path to the saved media file (e.g., '/media/uuid_filename.jpg')"
    )


class DetectionHistoryResponse(BaseModel):
    """
    Response model for detection history endpoint.
    
    Contains a list of recent detection records.
    """
    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 20,
                "records": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "file_name": "example1.jpg",
                        "file_type": "image",
                        "file_size": 2048576,
                        "detection_score": 0.92,
                        "classification": "Fake",
                        "model_version": "ImageDeepfakeNet-v1.0",
                        "timestamp": "2026-03-24T10:30:00Z",
                        "processing_duration": 1.234
                    },
                    {
                        "id": "223e4567-e89b-12d3-a456-426614174001",
                        "file_name": "example2.mp4",
                        "file_type": "video",
                        "file_size": 10485760,
                        "detection_score": 0.78,
                        "classification": "Real",
                        "model_version": "VideoDeepfakeNet-v1.0",
                        "timestamp": "2026-03-24T09:15:00Z",
                        "processing_duration": 12.567
                    }
                ]
            }
        }
    }
    
    total: int = Field(
        description="Total number of records returned"
    )
    
    records: List[Dict[str, Any]] = Field(
        description="List of detection records (newest first)"
    )


class PlatformStatsResponse(BaseModel):
    """
    Response model for platform statistics.
    
    Aggregated metrics about all detection scans.
    """
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_scans": 1547,
                "deepfakes_detected": 423,
                "real_media_detected": 1124,
                "avg_processing_duration": 2.345,
                "scans_by_type": {
                    "image": 892,
                    "video": 445,
                    "audio": 210
                },
                "classification_breakdown": {
                    "Real": 1124,
                    "Fake": 423
                }
            }
        }
    }
    
    total_scans: int = Field(
        description="Total number of scans performed"
    )
    
    deepfakes_detected: int = Field(
        description="Number of deepfakes detected"
    )
    
    real_media_detected: int = Field(
        description="Number of real media detected"
    )
    
    avg_processing_duration: float = Field(
        description="Average processing time in seconds"
    )
    
    scans_by_type: Dict[str, int] = Field(
        description="Breakdown of scans by media type"
    )
    
    classification_breakdown: Dict[str, int] = Field(
        description="Breakdown of classifications (Real vs Fake)"
    )
