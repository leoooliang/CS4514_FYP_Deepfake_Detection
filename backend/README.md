# Deepfake Detection System - Backend

## Overview

This folder contains the **FastAPI** application for the Final Year Project *Detection of Deepfake Images, Videos, and Audio*. It exposes HTTP endpoints that accept uploaded media, run the PyTorch-based detection pipeline, and return predictions together with persistence in **SQLite** where configured.

## Prerequisites

- **Python 3.10 or newer** (3.10+)
- **FFmpeg** installed and available on your `PATH` (used for video preprocessing)
- An **NVIDIA GPU with CUDA** support, recommended for practical inference latency when loading and running the PyTorch models (CPU-only execution is also supported)

## Environment Setup & Installation

1. **Create a virtual environment** (from this `backend` directory):

   ```bash
   cd backend
   python -m venv .venv
   ```

2. **Activate the virtual environment**

   - Windows (PowerShell):

     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```

   - macOS / Linux:

     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Model Weights & Database Initialization

**Pre-trained weights (Google Drive)**  
Due to file size limits, please download the **saved model weights** (`.pth` files) from the Google Drive ([Click here](https://drive.google.com/drive/folders/1wpxQEAisTJALgAd0Y4g-svjv8B94iMXN?usp=drive_link)) and **place** the files in `backend/models/`. (paths in `.env` or the defaults in `app/config.py` should match the filenames you use)

**Database**  
The application uses **SQLite** by default. The database file is created when the application starts and tables are initialised on startup; no separate migration command is required for a typical local run.

## Running the Server

From the **`backend`** directory, with the virtual environment activated:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will listen on all interfaces on port **8000** (adjust `--port` if that port is already in use).

## Calling the API

**Base URL** (default local run): `http://localhost:8000`  
**Versioned API prefix**: `/api/v1` — all predict and telemetry routes below are under `http://localhost:8000/api/v1/...`.

### Quick reference

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | API name, version, status |
| `GET` | `/health` | Liveness and loaded model status |
| `POST` | `/api/v1/predict/image` | Image upload → prediction |
| `POST` | `/api/v1/predict/video` | Video upload → prediction |
| `POST` | `/api/v1/predict/audio` | Audio upload → prediction |
| `GET` | `/api/v1/telemetry/history` | Recent runs (`session_id`, `limit` query params) |
| `GET` | `/api/v1/telemetry/results/{record_id}` | Single record by ID |
| `GET` | `/api/v1/telemetry/stats` | Aggregated platform stats |

### Prediction requests (`POST`)

Use **`multipart/form-data`**:

- **`file`** (required): the media file.
- **`session_id`** (optional): string for correlating runs in telemetry.

## API documentation

With the server running, open the interactive **Swagger UI** in a browser:

[http://localhost:8000/docs](http://localhost:8000/docs)

This page documents all endpoints and allows you to try requests directly against the running API.
