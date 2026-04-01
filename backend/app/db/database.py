"""
============================================================================
Database Configuration - SQLAlchemy Engine and Session Management
============================================================================
Sets up the SQLAlchemy engine, session factory, and declarative base.

Database: SQLite (local file)
Location: ./sql_app.db (in the backend root directory)

Author: Senior Full-Stack Engineer
Date: 2026-03-24
============================================================================
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from typing import Generator
from loguru import logger


# ============================================================================
# Database Configuration
# ============================================================================

# SQLite database URL
# The database file will be created in the backend root directory
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Create SQLAlchemy engine
# check_same_thread=False is needed for SQLite to work with FastAPI
# In production with PostgreSQL/MySQL, this argument is not needed
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False  # Set to True for SQL query logging (debugging)
)

# Create SessionLocal class for database sessions
# Each instance of SessionLocal will be a database session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base for ORM models
Base = declarative_base()


# ============================================================================
# Database Dependency
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency for FastAPI endpoints.
    
    This function creates a new SQLAlchemy session for each request
    and ensures it's properly closed after the request completes.
    
    Usage in FastAPI endpoints:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            # Use db session here
            pass
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# Database Initialization
# ============================================================================

def init_db() -> None:
    """
    Initialize the database by creating all tables.
    
    This function should be called on application startup.
    It creates all tables defined in the models if they don't exist.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        raise
