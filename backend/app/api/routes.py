"""
Central router - aggregates all endpoint sub-routers.

Endpoint: /api/v1/predict/image
Endpoint: /api/v1/predict/video
Endpoint: /api/v1/predict/audio
Endpoint: /api/v1/telemetry/history
Endpoint: /api/v1/telemetry/results/{record_id}
Endpoint: /api/v1/telemetry/stats
"""

from fastapi import APIRouter

from app.api.endpoints import audio, image, telemetry, video

api_router = APIRouter()

api_router.include_router(image.router, prefix="/predict", tags=["image"])
api_router.include_router(video.router, prefix="/predict", tags=["video"])
api_router.include_router(audio.router, prefix="/predict", tags=["audio"])
api_router.include_router(telemetry.router, prefix="/telemetry", tags=["telemetry"])
