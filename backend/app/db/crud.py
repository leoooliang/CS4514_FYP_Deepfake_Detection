"""
============================================================================
CRUD Operations - Database Query Functions
============================================================================
Provides Create, Read, Update, Delete operations for detection records.

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from loguru import logger

from app.db.models import DetectionRecord


# ============================================================================
# Create Operations
# ============================================================================

def create_detection_record(
    db: Session,
    record_data: Dict[str, Any]
) -> DetectionRecord:
    """
    Create a new detection record in the database.
    
    Args:
        db: SQLAlchemy database session
        record_data: Dictionary containing record fields:
            - file_name (str): Original filename
            - file_type (str): "image", "audio", or "video"
            - file_size (int): File size in bytes
            - detection_score (float): Confidence score (0.0-1.0)
            - classification (str): "Real" or "Fake"
            - model_version (str): Model identifier
            - processing_duration (float): Processing time in seconds
            - session_id (str, optional): Session ID for anonymous user tracking
            - media_path (str, optional): URL path to saved media file
    
    Returns:
        DetectionRecord: The created record with generated ID
    
    Raises:
        Exception: If database operation fails
    """
    try:
        # Create new record instance
        record = DetectionRecord(**record_data)
        
        # Add to session and commit
        db.add(record)
        db.commit()
        db.refresh(record)
        
        logger.info(
            f"✓ Detection record created: {record.id} | "
            f"Type: {record.file_type} | "
            f"Classification: {record.classification} | "
            f"Session: {record.session_id or 'N/A'}"
        )
        
        return record
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to create detection record: {str(e)}")
        raise


# ============================================================================
# Read Operations
# ============================================================================

def get_recent_history(
    db: Session,
    session_id: Optional[str] = None,
    limit: int = 20
) -> List[DetectionRecord]:
    """
    Get the most recent detection records, optionally filtered by session_id.
    
    Args:
        db: SQLAlchemy database session
        session_id: Optional session ID to filter records for a specific user
        limit: Maximum number of records to return (default: 20)
    
    Returns:
        List[DetectionRecord]: List of recent detection records,
                               ordered by timestamp (newest first)
    """
    try:
        query = db.query(DetectionRecord)
        
        # Filter by session_id if provided
        if session_id:
            query = query.filter(DetectionRecord.session_id == session_id)
        
        records = (
            query
            .order_by(desc(DetectionRecord.timestamp))
            .limit(limit)
            .all()
        )
        
        logger.debug(
            f"Retrieved {len(records)} recent detection records"
            f"{f' for session {session_id}' if session_id else ''}"
        )
        return records
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve recent history: {str(e)}")
        raise


def get_record_by_id(
    db: Session,
    record_id: str
) -> Optional[DetectionRecord]:
    """
    Get a specific detection record by its UUID.
    
    Args:
        db: SQLAlchemy database session
        record_id: UUID of the detection record
    
    Returns:
        DetectionRecord: The detection record, or None if not found
    """
    try:
        record = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.id == record_id)
            .first()
        )
        
        if record:
            logger.debug(f"Retrieved detection record: {record_id}")
        else:
            logger.warning(f"Detection record not found: {record_id}")
        
        return record
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve record {record_id}: {str(e)}")
        raise


def get_platform_stats(db: Session) -> Dict[str, Any]:
    """
    Get aggregated platform statistics.
    
    Calculates:
        - Total number of scans performed
        - Average processing duration
        - Count of deepfakes detected
        - Breakdown by media type
    
    Args:
        db: SQLAlchemy database session
    
    Returns:
        dict: Platform statistics including:
            - total_scans (int): Total number of detections
            - deepfakes_detected (int): Number of deepfakes found
            - avg_processing_duration (float): Average processing time in seconds
            - scans_by_type (dict): Breakdown by media type
            - classification_breakdown (dict): Counts of Real vs Fake
    """
    try:
        # Total scans
        total_scans = db.query(func.count(DetectionRecord.id)).scalar() or 0
        
        # Deepfakes detected (classification = "Fake")
        deepfakes_detected = (
            db.query(func.count(DetectionRecord.id))
            .filter(DetectionRecord.classification == "Fake")
            .scalar() or 0
        )
        
        # Average processing duration
        avg_processing_duration = (
            db.query(func.avg(DetectionRecord.processing_duration))
            .scalar() or 0.0
        )
        
        # Scans by type
        scans_by_type_raw = (
            db.query(
                DetectionRecord.file_type,
                func.count(DetectionRecord.id).label("count")
            )
            .group_by(DetectionRecord.file_type)
            .all()
        )
        
        scans_by_type = {
            file_type: count for file_type, count in scans_by_type_raw
        }
        
        # Classification breakdown
        classification_breakdown_raw = (
            db.query(
                DetectionRecord.classification,
                func.count(DetectionRecord.id).label("count")
            )
            .group_by(DetectionRecord.classification)
            .all()
        )
        
        classification_breakdown = {
            classification: count 
            for classification, count in classification_breakdown_raw
        }
        
        stats = {
            "total_scans": total_scans,
            "deepfakes_detected": deepfakes_detected,
            "real_media_detected": classification_breakdown.get("Real", 0),
            "avg_processing_duration": round(avg_processing_duration, 3),
            "scans_by_type": scans_by_type,
            "classification_breakdown": classification_breakdown
        }
        
        logger.debug(f"Platform stats retrieved: {total_scans} total scans")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve platform stats: {str(e)}")
        raise


# ============================================================================
# Additional Query Functions
# ============================================================================

def get_records_by_type(
    db: Session,
    file_type: str,
    limit: int = 20
) -> List[DetectionRecord]:
    """
    Get recent detection records filtered by media type.
    
    Args:
        db: SQLAlchemy database session
        file_type: Media type ("image", "audio", or "video")
        limit: Maximum number of records to return
    
    Returns:
        List[DetectionRecord]: Filtered detection records
    """
    try:
        records = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.file_type == file_type)
            .order_by(desc(DetectionRecord.timestamp))
            .limit(limit)
            .all()
        )
        
        logger.debug(
            f"Retrieved {len(records)} {file_type} detection records"
        )
        return records
        
    except Exception as e:
        logger.error(
            f"❌ Failed to retrieve {file_type} records: {str(e)}"
        )
        raise


def get_deepfake_records(
    db: Session,
    limit: int = 20
) -> List[DetectionRecord]:
    """
    Get recent records where deepfakes were detected.
    
    Args:
        db: SQLAlchemy database session
        limit: Maximum number of records to return
    
    Returns:
        List[DetectionRecord]: Records classified as deepfakes
    """
    try:
        records = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.classification == "Fake")
            .order_by(desc(DetectionRecord.timestamp))
            .limit(limit)
            .all()
        )
        
        logger.debug(f"Retrieved {len(records)} deepfake records")
        return records
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve deepfake records: {str(e)}")
        raise
