"""
============================================================================
Telemetry and History Endpoints
============================================================================
API endpoints for retrieving detection history and platform statistics.

Endpoints:
    - GET /api/v1/history: Get recent detection history
    - GET /api/v1/stats: Get platform statistics
    - GET /api/v1/results/{id}: Get specific detection result

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from loguru import logger

from app.db.database import get_db
from app.db import crud
from app.schemas.telemetry import (
    DetectionHistoryResponse,
    DetectionRecordResponse,
    PlatformStatsResponse
)


# Create router for telemetry endpoints
router = APIRouter()


# =============================================================================
# History Endpoints
# =============================================================================

@router.get(
    "/history",
    response_model=DetectionHistoryResponse,
    summary="Get recent detection history",
    description="""
    Retrieve the most recent detection records from the database.
    
    Returns metadata and results from recent scans, ordered by timestamp
    (newest first). Does NOT include actual media files or file hashes.
    
    **Use cases:**
    - Display recent scan history in the frontend
    - Track detection trends
    - Audit detection activities
    
    **Privacy:** Only metadata is stored, not the actual media files.
    """,
    responses={
        200: {
            "description": "Successfully retrieved history",
            "content": {
                "application/json": {
                    "example": {
                        "total": 20,
                        "records": [
                            {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "file_name": "example.jpg",
                                "file_type": "image",
                                "file_size": 2048576,
                                "detection_score": 0.92,
                                "classification": "Fake",
                                "model_version": "ImageDeepfakeNet-v1.0",
                                "timestamp": "2026-03-24T10:30:00Z",
                                "processing_duration": 1.234
                            }
                        ]
                    }
                }
            }
        },
        500: {"description": "Database error"}
    },
    tags=["telemetry"]
)
async def get_detection_history(
    session_id: Optional[str] = Query(None, description="Session ID to filter history for a specific user"),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of records to return (1-100)"
    ),
    db: Session = Depends(get_db)
) -> DetectionHistoryResponse:
    """
    Get the most recent detection records, optionally filtered by session_id.
    
    Args:
        session_id: Optional session ID to filter records for a specific user
        limit: Maximum number of records to return
        db: Database session (injected by FastAPI)
    
    Returns:
        DetectionHistoryResponse: List of recent detection records
    
    Raises:
        HTTPException: If database query fails
    """
    try:
        logger.info(
            f"Retrieving detection history (limit={limit}"
            f"{f', session_id={session_id}' if session_id else ''})"
        )
        
        # Query recent records with optional session_id filter
        records = crud.get_recent_history(db, session_id=session_id, limit=limit)
        
        # Convert to dict format
        records_data = [record.to_dict() for record in records]
        
        return DetectionHistoryResponse(
            total=len(records_data),
            records=records_data
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve detection history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detection history"
        )


@router.get(
    "/results/{record_id}",
    response_model=DetectionRecordResponse,
    summary="Get specific detection result",
    description="""
    Retrieve a specific detection record by its UUID.
    
    Use this endpoint to fetch details about a previously performed detection.
    The record ID is returned in the initial detection response.
    """,
    responses={
        200: {
            "description": "Successfully retrieved record",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "file_name": "example.jpg",
                        "file_type": "image",
                        "file_size": 2048576,
                        "detection_score": 0.92,
                        "classification": "Fake",
                        "model_version": "ImageDeepfakeNet-v1.0",
                        "timestamp": "2026-03-24T10:30:00Z",
                        "processing_duration": 1.234
                    }
                }
            }
        },
        404: {"description": "Record not found"},
        500: {"description": "Database error"}
    },
    tags=["telemetry"]
)
async def get_detection_result(
    record_id: str,
    db: Session = Depends(get_db)
) -> DetectionRecordResponse:
    """
    Get a specific detection record by ID.
    
    Args:
        record_id: UUID of the detection record
        db: Database session (injected by FastAPI)
    
    Returns:
        DetectionRecordResponse: The requested detection record
    
    Raises:
        HTTPException: If record not found or database query fails
    """
    try:
        logger.info(f"Retrieving detection record: {record_id}")
        
        # Query record by ID
        record = crud.get_record_by_id(db, record_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Detection record not found: {record_id}"
            )
        
        # Convert to dict and return
        return DetectionRecordResponse(**record.to_dict())
        
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to retrieve detection record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detection record"
        )


# =============================================================================
# Statistics Endpoints
# =============================================================================

@router.get(
    "/stats",
    response_model=PlatformStatsResponse,
    summary="Get platform statistics",
    description="""
    Retrieve aggregated statistics about all detection scans.
    
    Includes:
    - Total number of scans performed
    - Count of deepfakes detected vs real media
    - Average processing duration
    - Breakdown by media type (image/audio/video)
    
    **Use cases:**
    - Display platform usage metrics
    - Monitor system performance
    - Generate analytics dashboards
    """,
    responses={
        200: {
            "description": "Successfully retrieved statistics",
            "content": {
                "application/json": {
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
        },
        500: {"description": "Database error"}
    },
    tags=["telemetry"]
)
async def get_platform_statistics(
    db: Session = Depends(get_db)
) -> PlatformStatsResponse:
    """
    Get aggregated platform statistics.
    
    Args:
        db: Database session (injected by FastAPI)
    
    Returns:
        PlatformStatsResponse: Aggregated statistics
    
    Raises:
        HTTPException: If database query fails
    """
    try:
        logger.info("Retrieving platform statistics")
        
        # Query aggregated stats
        stats = crud.get_platform_stats(db)
        
        return PlatformStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to retrieve platform statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve platform statistics"
        )
