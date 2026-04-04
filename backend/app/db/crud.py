"""
CRUD helpers for detection records.

Helper functions:
- create_detection_record
- get_recent_history
- get_record_by_id
- get_platform_stats
- get_records_by_type
- get_deepfake_records
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db.models import DetectionRecord


def create_detection_record(db: Session, record_data: Dict[str, Any]) -> DetectionRecord:
    try:
        record = DetectionRecord(**record_data)
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(
            "[DB] Detection record created: id={}, file='{}', type={}, "
            "classification={}, score={:.4f}, duration={:.3f}s",
            record.id, record.file_name, record.file_type,
            record.classification, record.detection_score,
            record.processing_duration,
        )
        return record
    except Exception as e:
        db.rollback()
        logger.error("[DB] Failed to create detection record: {}", e)
        raise


def get_recent_history(
    db: Session, session_id: Optional[str] = None, limit: int = 20
) -> List[DetectionRecord]:
    query = db.query(DetectionRecord)
    if session_id:
        query = query.filter(DetectionRecord.session_id == session_id)
    return query.order_by(desc(DetectionRecord.timestamp)).limit(limit).all()


def get_record_by_id(db: Session, record_id: str) -> Optional[DetectionRecord]:
    return db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()


def get_platform_stats(db: Session) -> Dict[str, Any]:
    total_scans = db.query(func.count(DetectionRecord.id)).scalar() or 0
    deepfakes_detected = (
        db.query(func.count(DetectionRecord.id))
        .filter(DetectionRecord.classification.in_(["Fake", "Deepfake"]))
        .scalar()
        or 0
    )
    avg_duration = db.query(func.avg(DetectionRecord.processing_duration)).scalar() or 0.0

    scans_by_type = {
        ft: cnt
        for ft, cnt in db.query(
            DetectionRecord.file_type, func.count(DetectionRecord.id)
        )
        .group_by(DetectionRecord.file_type)
        .all()
    }
    raw_breakdown = {
        cls: cnt
        for cls, cnt in db.query(
            DetectionRecord.classification, func.count(DetectionRecord.id)
        )
        .group_by(DetectionRecord.classification)
        .all()
    }
    merged_deepfake = raw_breakdown.get("Fake", 0) + raw_breakdown.get("Deepfake", 0)
    classification_breakdown = {k: v for k, v in raw_breakdown.items() if k not in ("Fake", "Deepfake")}
    if merged_deepfake:
        classification_breakdown["Deepfake"] = merged_deepfake

    return {
        "total_scans": total_scans,
        "deepfakes_detected": deepfakes_detected,
        "real_media_detected": raw_breakdown.get("Real", 0),
        "avg_processing_duration": round(avg_duration, 3),
        "scans_by_type": scans_by_type,
        "classification_breakdown": classification_breakdown,
    }


def get_records_by_type(db: Session, file_type: str, limit: int = 20) -> List[DetectionRecord]:
    return (
        db.query(DetectionRecord)
        .filter(DetectionRecord.file_type == file_type)
        .order_by(desc(DetectionRecord.timestamp))
        .limit(limit)
        .all()
    )


def get_deepfake_records(db: Session, limit: int = 20) -> List[DetectionRecord]:
    return (
        db.query(DetectionRecord)
        .filter(DetectionRecord.classification.in_(["Fake", "Deepfake"]))
        .order_by(desc(DetectionRecord.timestamp))
        .limit(limit)
        .all()
    )
