# Deepfake Detection Module Training

This folder contains all training, evaluation, and data preparation code for the deepfake detection module developed for the Final Year Project. Three detection modules are implemented: 
- **Image Detection Module** (image detector)
- **Audio Detection Module** (audio detector)
- **Video Detection Module** (video detector).

---

## Folder Structure

```
model_training/
├── common/               # GPU augmentations, checkpoint I/O, plotting functions
├── configs/              # Training configurations for all modules.
├── data/                 # Processed datasets used by the notebooks
├── data_loaders/         # Custom Dataset classes for each module
├── data_process_scripts/ # Scripts to prepare raw datasets
├── engine/               # Shared training loop, evaluation, and early stopping logic
├── models/               # Model definitions for image, audio, video detectors
├── notebooks/            # Jupyter notebooks for training and evaluation of modules
├── raw_data/             # Raw datasets 
├── report_figures/       # Scripts and generated figures for the final report
└── saved_models/         # Saved model weights 
```

---

## Environment Setup & Installation

### Prerequisites

- **Python 3.10+**
- **NVIDIA GPU with CUDA support** — training relies on mixed-precision (AMP) and GPU-accelerated augmentations; a CUDA-capable GPU is required.
- **FFmpeg** — needed by the video preprocessing script. Install via your package manager:
  - Windows: `choco install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

### Installing Dependencies

A `requirements.txt` is provided at the repository root under `backend/`. The core deep-learning dependencies relevant to model training are listed below. Install the full backend dependency set (which covers all training requirements):

```bash
pip install -r backend/requirements.txt
pip install kornia scikit-learn matplotlib seaborn jupyter
```

> **Note:** PyTorch with CUDA 12.1 is used. If your system has a different CUDA version, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct `--index-url`.

---

## Dataset Preparation

Raw datasets must be downloaded separately and placed under `model_training/raw_data/` before running the preprocessing scripts. The expected layout is:

```
model_training/raw_data/
├── ArtiFact/                       # ArtiFact image dataset
│   ├── real/   (ffhq, celebahq)
│   └── fake/   (stable-diffusion, stylegan, stargan, sfhq, face_synthetics)
├── FaceForensics++_C23/            # FaceForensics++ (C23 compression)
│   ├── original/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   ├── NeuralTextures/
│   └── FaceShifter/
├── ASVspoof2021_DF_eval/           # ASVspoof 2021 DF evaluation set
│   ├── trial_metadata.txt
│   └── flac/
└── FakeAVCeleb_v1.2/              # FakeAVCeleb v1.2 video dataset
    └── meta_data.csv + video dirs
```

Once the raw data is in place, run each preprocessing script from the **repository root**:

```bash
# Image data — extract and crop faces from ArtiFact and FaceForensics++
python -m model_training.data_process_scripts.process_artifact_dataset
python -m model_training.data_process_scripts.process_faceforensics_to_images

# Audio data — sample and split ASVspoof 2021 into class folders
python -m model_training.data_process_scripts.prepare_audio_dataset

# Video data — extract face frames + audio waveforms from FakeAVCeleb video into .pt tensors
python -m model_training.data_process_scripts.preprocess_video
```

After preprocessing, the processed datasets will be written to `model_training/data/` with `train/`, `val/`, and `test/` splits under each modality subfolder (`image/`, `audio/`, `video_tensors/`).

---

## How to Run the Code

Training and evaluation are driven through **Jupyter notebooks** located in `model_training/notebooks/`. Each notebook imports from the shared `configs`, `models`, `data_loaders`, `engine`, and `common` packages.

Open and run the relevant notebook **cell by cell** (or "Run All"):

| Notebook | Detection Module | Description |
|----------|------------------|-------------|
| `image_CLIP_Stream.ipynb` | Image — Spatial Stream | Trains the CLIP ViT-L/14 classifier |
| `image_Noise_Stream.ipynb` | Image — Noise Stream | Trains the SRM + EfficientNetV2-S  |
| `image_two_stream_fusion.ipynb` | Image — Two-Stream Fusion | Evaluates spatial (CLIP) + noise (SRM/ENetV2) stream fusion models  |
| `audio_CNN_GRU.ipynb` | Audio | Trains the dual-feature (Mel + LFCC) CNN-GRU  |
| `video_multimodal_network.ipynb` | Video | Trains the tri-stream (image + audio + sync) detection network |

Saved model weights (`.pt` files) should be placed in `model_training/saved_models/` .

Note: The `model_training/saved_models/` folder is empty when cloned from GitHub. All required model weights (`.pt` files) are stored in `backend/models/`.

---

## Important Note for Assessors

Due to the random weight initialization, GPU non-determinism in cuDNN operations, and mixed-precision arithmetic, running the training notebooks from scratch may produce metrics (e.g., accuracy, AUC-ROC) that differ slightly from the exact figures reported in the Final Report. However, the overall convergence behaviour and relative performance across modules should remain consistent. The saved model checkpoints used to generate the reported results are included where possible to allow direct reproduction of the evaluation metrics without retraining.
