"""FastAPI application entry point."""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import psutil
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.routes import api_router
from app.config import settings
from app.core.request_context import bind_request_id, reset_request_id
from app.db.database import init_db
from app.models.factory import DetectorFactory


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("STARTUP - Multimodal Deepfake Detection System v{}", settings.VERSION)
    logger.info("Environment: {} | Debug: {}", settings.ENVIRONMENT, settings.DEBUG)
    logger.info("Host: {}:{}", settings.HOST, settings.PORT)

    persistent_media_dir = os.path.join(os.getcwd(), "persistent_media")
    os.makedirs(persistent_media_dir, exist_ok=True)
    logger.info("Persistent media directory: {}", persistent_media_dir)

    logger.info("Initialising database...")
    init_db()

    if not getattr(app.state, "detector_factory", None):
        logger.info("Loading ML detector models...")
        start = time.time()
        factory = DetectorFactory()
        for dt in ("image", "video", "audio"):
            t0 = time.time()
            factory.get_detector(dt)
            logger.info(
                "[MODEL:{}] detector loaded in {:.2f}s", dt.upper(), time.time() - t0
            )
        elapsed = time.time() - start
        logger.info("All models loaded successfully in {:.2f}s", elapsed)
        app.state.detector_factory = factory

    logger.info("System ready - accepting requests")

    yield

    logger.info("SHUTDOWN - Cleaning up resources...")
    if hasattr(app.state, "detector_factory"):
        del app.state.detector_factory
        logger.info("Detector models released")
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal Deepfake Detection API",
    description="Detect manipulated images, videos, and audio using deep learning.",
    version=settings.VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Health check and system status"},
        {"name": "image", "description": "Image deepfake detection"},
        {"name": "video", "description": "Video deepfake detection"},
        {"name": "audio", "description": "Audio deepfake detection"},
        {"name": "telemetry", "description": "Detection history and statistics"},
    ],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)

if settings.ENVIRONMENT == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)


@app.middleware("http")
async def request_id_and_process_time(request: Request, call_next):
    rid, token = bind_request_id()
    start = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time"] = str(round(time.time() - start, 3))
        return response
    finally:
        reset_request_id(token)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "message": "Invalid request data", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on {} {}: {}", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Static files & routes
# ---------------------------------------------------------------------------

_media_dir = os.path.join(os.getcwd(), "persistent_media")
os.makedirs(_media_dir, exist_ok=True)
app.mount("/media", StaticFiles(directory=_media_dir), name="media")

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"], summary="Root endpoint")
async def root() -> Dict[str, Any]:
    return {
        "message": "Multimodal Deepfake Detection API",
        "version": settings.VERSION,
        "status": "operational",
    }


@app.get("/health", tags=["health"], summary="Health check")
async def health_check(request: Request) -> Dict[str, Any]:
    try:
        factory = request.app.state.detector_factory
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models": {dt: info.get("status", "unknown") for dt, info in factory.get_loaded_detectors().items()},
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
            },
        }
    except Exception as e:
        logger.error("Health check failed: {}", e)
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"status": "unhealthy", "error": str(e)})


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
