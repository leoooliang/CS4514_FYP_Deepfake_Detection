"""
============================================================================
Database Package - SQLAlchemy ORM Layer
============================================================================
This package provides database models, CRUD operations, and session management
for the deepfake detection telemetry system.

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from app.db.database import Base, engine, SessionLocal, get_db

__all__ = ["Base", "engine", "SessionLocal", "get_db"]
