"""
SQLAlchemy ORM model for detection records.
"""

import uuid

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db.database import Base


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        nullable=False,
    )
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False, index=True)
    file_size = Column(Integer, nullable=False)
    detection_score = Column(Float, nullable=False)
    classification = Column(String(10), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    processing_duration = Column(Float, nullable=False)
    session_id = Column(String(255), nullable=True, index=True)
    media_path = Column(String(512), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<DetectionRecord(id='{self.id}', type='{self.file_type}', "
            f"classification='{self.classification}', score={self.detection_score:.3f})>"
        )

    def to_dict(self) -> dict:
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
            "media_path": self.media_path,
        }
