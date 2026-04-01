"""
============================================================================
API Routes - Central Router Configuration
============================================================================
This module aggregates all endpoint routers and creates the main API router.

Author: Senior Full-Stack Engineer
Date: 2026-01-28
============================================================================
"""

from fastapi import APIRouter

from app.api.endpoints import image, video, audio, telemetry


# Create main API router
api_router = APIRouter()

# Include endpoint routers
# Each router is responsible for its own endpoints
api_router.include_router(
    image.router,
    tags=["image"]
)

api_router.include_router(
    video.router,
    tags=["video"]
)

api_router.include_router(
    audio.router,
    tags=["audio"]
)

api_router.include_router(
    telemetry.router,
    prefix="/telemetry",
    tags=["telemetry"]
)
