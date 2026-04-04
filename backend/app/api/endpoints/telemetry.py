"""
Telemetry endpoints for detection history and platform statistics.

Endpoint: /api/v1/telemetry/history
Response model: DetectionHistoryResponse
Tags: telemetry

Endpoint: /api/v1/telemetry/results/{record_id}
Response model: DetectionRecordResponse
Tags: telemetry

Endpoint: /api/v1/telemetry/stats
Response model: PlatformStatsResponse
Tags: telemetry
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.schemas.telemetry import DetectionHistoryResponse, DetectionRecordResponse, PlatformStatsResponse

router = APIRouter()


@router.get("/history", response_model=DetectionHistoryResponse, summary="Recent detection history", tags=["telemetry"])
async def get_detection_history(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> DetectionHistoryResponse:
    try:
        records = crud.get_recent_history(db, session_id=session_id, limit=limit)
        return DetectionHistoryResponse(total=len(records), records=[r.to_dict() for r in records])
    except Exception as e:
        logger.error("[TELEMETRY] Failed to retrieve history (session_id={}): {}", session_id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to retrieve detection history")


@router.get("/results/{record_id}", response_model=DetectionRecordResponse, summary="Get detection result by ID", tags=["telemetry"])
async def get_detection_result(record_id: str, db: Session = Depends(get_db)) -> DetectionRecordResponse:
    try:
        record = crud.get_record_by_id(db, record_id)
        if not record:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Record not found: {record_id}")
        return DetectionRecordResponse(**record.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[TELEMETRY] Failed to retrieve record '{}': {}", record_id, e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to retrieve record")


@router.get("/stats", response_model=PlatformStatsResponse, summary="Platform statistics", tags=["telemetry"])
async def get_platform_statistics(db: Session = Depends(get_db)) -> PlatformStatsResponse:
    try:
        return PlatformStatsResponse(**crud.get_platform_stats(db))
    except Exception as e:
        logger.error("[TELEMETRY] Failed to retrieve platform stats: {}", e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to retrieve platform statistics")
