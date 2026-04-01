"""
============================================================================
Database Models - SQLAlchemy ORM Models
============================================================================
Defines the database schema for detection records.

Privacy Notice:
    This database stores ONLY metadata and detection results.
    It does NOT store actual media files or file hashes to ensure user privacy.

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from app.db.database import Base


class DetectionRecord(Base):
    """
    SQLAlchemy model for storing deepfake detection results.
    
    This table stores metadata and results from each detection scan.
    It does NOT store the actual media files or file hashes.
    
    Columns:
        id: Unique identifier (UUID4)
        file_name: Original filename (sanitized)
        file_type: Type of media ("image", "audio", "video")
        file_size: File size in bytes
        detection_score: Confidence score (0.0 to 1.0)
        classification: Result classification ("Real" or "Fake")
        model_version: Version of the model used
        timestamp: When the detection was performed (UTC)
        processing_duration: Time taken to process (in seconds)
    """
    
    __tablename__ = "detection_records"
    
    # Primary key - UUID for global uniqueness
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        nullable=False,
        doc="Unique identifier for the detection record"
    )
    
    # File metadata (no actual content or hash)
    file_name = Column(
        String(255),
        nullable=False,
        doc="Original filename (sanitized)"
    )
    
    file_type = Column(
        String(10),
        nullable=False,
        index=True,
        doc="Media type: 'image', 'audio', or 'video'"
    )
    
    file_size = Column(
        Integer,
        nullable=False,
        doc="File size in bytes"
    )
    
    # Detection results
    detection_score = Column(
        Float,
        nullable=False,
        doc="Confidence score from 0.0 to 1.0"
    )
    
    classification = Column(
        String(10),
        nullable=False,
        index=True,
        doc="Classification result: 'Real' or 'Fake'"
    )
    
    # Model information
    model_version = Column(
        String(50),
        nullable=False,
        doc="Version or identifier of the ML model used"
    )
    
    # Timing information
    timestamp = Column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False,
        index=True,
        doc="Timestamp when the detection was performed (UTC)"
    )
    
    processing_duration = Column(
        Float,
        nullable=False,
        doc="Processing time in seconds"
    )
    
    # Session tracking for anonymous users
    session_id = Column(
        String(255),
        nullable=True,
        index=True,
        doc="Session ID to track anonymous user history"
    )
    
    # Persistent media storage
    media_path = Column(
        String(512),
        nullable=True,
        doc="URL path to the saved media file (e.g., '/media/uuid_filename.mp4')"
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<DetectionRecord(id='{self.id}', "
            f"type='{self.file_type}', "
            f"classification='{self.classification}', "
            f"score={self.detection_score:.3f})>"
        )
    
    def to_dict(self) -> dict:
        """
        Convert record to dictionary for JSON serialization.
        
        Returns:
            dict: Record data as dictionary
        """
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "detection_score": self.detection_score,
            "classification": self.classification,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "processing_duration": self.processing_duration,
            "session_id": self.session_id,
            "media_path": self.media_path
        }
