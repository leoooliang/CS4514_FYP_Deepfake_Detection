"""
============================================================================
Multimodal Deepfake Detection System - FastAPI Main Application
============================================================================
This is the entry point for the FastAPI backend server.

Architecture:
    - Microservices-ready design with decoupled components
    - Factory pattern for ML model management
    - Async request handling for high performance
    - CORS enabled for frontend communication

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.api.routes import api_router
from app.models.factory import DetectorFactory
from app.db.database import init_db


# ============================================================================
# Application Lifespan Management
# ============================================================================
# This function manages startup and shutdown events for the application.
# It's responsible for loading ML models once and keeping them in memory.
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events (startup/shutdown).
    
    On Startup:
        - Initialize ML model factory
        - Load detector models into memory
        - Set up logging configuration
        - Warm up models with dummy predictions
    
    On Shutdown:
        - Clean up resources
        - Unload models from memory
        - Close any open connections
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Multimodal Deepfake Detection System")
    logger.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # STARTUP: Initialize Persistent Media Directory
    # -------------------------------------------------------------------------
    try:
        logger.info("📁 Initializing persistent media directory...")
        persistent_media_dir = os.path.join(os.getcwd(), "persistent_media")
        os.makedirs(persistent_media_dir, exist_ok=True)
        logger.info(f"✅ Persistent media directory ready: {persistent_media_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create persistent media directory: {str(e)}")
        raise
    
    # -------------------------------------------------------------------------
    # STARTUP: Initialize Database
    # -------------------------------------------------------------------------
    try:
        logger.info("🗄️ Initializing database...")
        init_db()
        logger.info("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        raise
    
    # -------------------------------------------------------------------------
    # STARTUP: Initialize ML Models
    # -------------------------------------------------------------------------
    # Check if detector factory is already set (e.g., by tests with mocks)
    if not hasattr(app.state, 'detector_factory') or app.state.detector_factory is None:
        try:
            logger.info("📦 Loading ML models...")
            start_time = time.time()
            
            # Initialize the detector factory with model configurations
            # The factory pattern allows us to easily swap models later
            factory = DetectorFactory()
            
            # Pre-load all detector models for faster inference
            # Note: In production, you might want to lazy-load based on usage patterns
            factory.get_detector("image")
            logger.info("✓ Image detector loaded")
            
            factory.get_detector("video")
            logger.info("✓ Video detector loaded")
            
            factory.get_detector("audio")
            logger.info("✓ Audio detector loaded")
            
            elapsed = time.time() - start_time
            logger.info(f"✅ All models loaded successfully in {elapsed:.2f}s")
            
            # Store factory in app state for access in route handlers
            app.state.detector_factory = factory
            
            logger.info("🌐 Server is ready to accept requests")
            logger.info(f"📍 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {str(e)}")
            raise
    else:
        logger.info("✓ Using pre-configured detector factory (test mode)")
    
    # -------------------------------------------------------------------------
    # APPLICATION RUNNING
    # -------------------------------------------------------------------------
    yield
    
    # -------------------------------------------------------------------------
    # SHUTDOWN: Cleanup Resources
    # -------------------------------------------------------------------------
    logger.info("🛑 Shutting down gracefully...")
    
    try:
        # Clean up any temporary files
        if hasattr(app.state, 'detector_factory'):
            # Unload models to free memory
            del app.state.detector_factory
            logger.info("✓ Models unloaded from memory")
        
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"⚠️ Error during shutdown: {str(e)}")


# ============================================================================
# FastAPI Application Instance
# ============================================================================

app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="""
    🔍 **Advanced Deepfake Detection System**
    
    A state-of-the-art API for detecting manipulated media across multiple modalities:
    - 🖼️ **Images**: Detect face swaps, GAN-generated images, and manipulations
    - 🎥 **Videos**: Frame-by-frame analysis with temporal consistency checks
    - 🎵 **Audio**: Voice cloning and speech synthesis detection
    
    ## Features
    - Real-time inference with GPU acceleration
    - Confidence scores and detailed metadata
    - Support for multiple file formats
    - RESTful API design
    - Comprehensive error handling
    
    ## Rate Limits
    - Image: 10 MB max file size
    - Video: 100 MB max file size
    - Audio: 10 MB max file size
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "image",
            "description": "Image deepfake detection operations"
        },
        {
            "name": "video",
            "description": "Video deepfake detection operations"
        },
        {
            "name": "audio",
            "description": "Audio deepfake detection operations"
        },
        {
            "name": "telemetry",
            "description": "Detection history and platform statistics"
        }
    ]
)


# ============================================================================
# Middleware Configuration
# ============================================================================

# -----------------------------------------------------------------------------
# CORS Middleware - Enable cross-origin requests from frontend
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# -----------------------------------------------------------------------------
# Trusted Host Middleware - Prevent Host Header attacks
# -----------------------------------------------------------------------------
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )


# ============================================================================
# Global Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with user-friendly messages.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unexpected errors.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


# ============================================================================
# Request/Response Middleware
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time to response headers for performance monitoring.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


# ============================================================================
# Static Files Configuration
# ============================================================================

# Mount persistent media directory for serving uploaded files
persistent_media_path = os.path.join(os.getcwd(), "persistent_media")
os.makedirs(persistent_media_path, exist_ok=True)

app.mount(
    "/media",
    StaticFiles(directory=persistent_media_path),
    name="media"
)

logger.info(f"📂 Static media files mounted at /media -> {persistent_media_path}")


# ============================================================================
# Route Registration
# ============================================================================

# Include all API routes with version prefix
app.include_router(
    api_router,
    prefix=settings.API_V1_PREFIX
)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["health"],
    summary="Root endpoint",
    description="Welcome message and API information"
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint - provides basic API information and links.
    """
    return {
        "message": "🔍 Multimodal Deepfake Detection API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": f"http://{settings.HOST}:{settings.PORT}/docs",
        "endpoints": {
            "image_detection": f"{settings.API_V1_PREFIX}/predict/image",
            "video_detection": f"{settings.API_V1_PREFIX}/predict/video",
            "audio_detection": f"{settings.API_V1_PREFIX}/predict/audio",
        }
    }


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API and ML models are operational"
)
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Health check endpoint - verify system status and model availability.
    
    Returns:
        dict: System health status including model availability
    """
    try:
        # Check if models are loaded
        factory = request.app.state.detector_factory
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models": {
                "image_detector": "loaded",
                "video_detector": "loaded",
                "audio_detector": "loaded"
            },
            "system": {
                "cpu_percent": __import__("psutil").cpu_percent(),
                "memory_percent": __import__("psutil").virtual_memory().percent
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting application in development mode")
    
    # Run the application with uvicorn
    # For production, use: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,  # Auto-reload on code changes (dev only)
        log_level="info"
    )
