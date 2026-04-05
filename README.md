# Detection of Deepfake Images, Videos, and Audio

## Overview

A deepfake detection **web application**: users upload images, audio, or video, and the system returns model predictions through a browser UI backed by a FastAPI service.

## Repository structure

| Folder | What it is |
|--------|------------|
| **`backend/`** | REST API: accepts uploads, runs PyTorch inference |
| **`frontend/`** | Single-page app: uploads, results, history | 
| **`model_training/`** | Training, evaluation, data preparation, and notebooks for the three detection modules |

**Detailed setup** (dependencies, env, commands, model paths, datasets) lives in each folder’s own README:

- [backend/README.md](backend/README.md)
- [frontend/README.md](frontend/README.md)
- [model_training/README.md](model_training/README.md)

## Quick start (local)

You typically run three things in parallel:

1. **Backend** — Python venv, install `requirements.txt`, place trained **`.pth`** weights under `backend/models/`, then start **Uvicorn** using port **8000**.
2. **Frontend** — `npm install` and **`npm run dev`**, access web app with `http://localhost:5173`.
3. **Models** — Direcly run the notebooks for training and evaluating detection modules, or directly download [pre-trained model weights](https://drive.google.com/drive/folders/1wpxQEAisTJALgAd0Y4g-svjv8B94iMXN?usp=drive_link) for skipping the training.

> **Notes**: **FFmpeg** on `PATH` for video handling; a **CUDA** GPU is recommended for inference latency.

## Author

Developed by **LIANG Wai Ching** as a Final Year Project at the **Department of Computer Science, City University of Hong Kong**.
