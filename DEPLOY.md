# Running in Docker

## How it works

The models (DISK, LightGlue) are **fully frozen and never retrained**.
The `add` command runs only forward inference and saves feature vectors to `database/`.
The `database/` folder is the entire system memory about wafers; model weights never change.

The image is built **once on a machine with fast internet** and transferred as a file.
No internet is needed on the lab PC.

---

## Part A — Build the image

```bash
# 1. Build the image (downloads torch, kornia, model weights — ~5–8 GB)
docker build -t whatthewafer .

# 2. Export to an archive
docker save whatthewafer | gzip > whatthewafer.tar.gz
# Resulting file: ~5–7 GB — transfer via USB or network
```

---

## Part B — Deploy on the lab PC (Windows 10, no internet)

### One-time setup

1. **Docker Desktop** — install from [docs.docker.com/desktop/install/windows-install](https://docs.docker.com/desktop/install/windows-install/)
   - Enable WSL2 during installation (offered automatically)
   - After installation: Settings → Resources → WSL Integration → enable your distro

2. **NVIDIA driver** — version ≥ 522.06 (required for CUDA in containers via WSL2)
   - Download in advance from [nvidia.com/drivers](https://www.nvidia.com/drivers) on a machine with internet

3. **Load the image** (no internet needed):
```powershell
docker load -i whatthewafer.tar.gz
```

### Usage

```powershell
# Add a wafer to the database
docker run --gpus all `
    -v ${PWD}/wafers:/app/wafers `
    -v ${PWD}/database:/app/database `
    whatthewafer add wafers/wafer1_1.tif --new wafer1

# Add another view of an existing wafer
docker run --gpus all `
    -v ${PWD}/wafers:/app/wafers `
    -v ${PWD}/database:/app/database `
    whatthewafer add wafers/wafer1_2.tif --wafer wafer1

# Identify an unknown image
docker run --gpus all `
    -v ${PWD}/wafers:/app/wafers `
    -v ${PWD}/database:/app/database `
    whatthewafer query wafers/unknown.tif

# Visually compare two images
docker run --gpus all `
    -v ${PWD}/wafers:/app/wafers `
    -v ${PWD}/database:/app/database `
    -v ${PWD}:/app/out `
    whatthewafer compare wafers/a.tif wafers/b.tif --output out/result.png

# List all wafers in the database
docker run -v ${PWD}/database:/app/database whatthewafer list

# Without GPU (if --gpus is unavailable or the card is not detected)
docker run `
    -v ${PWD}/wafers:/app/wafers `
    -v ${PWD}/database:/app/database `
    whatthewafer --no-gpu query wafers/unknown.tif
```

### Via docker compose (more convenient for regular use)

```powershell
# First run builds the image if not present
docker compose run --rm whatthewafer query wafers/unknown.tif
docker compose run --rm whatthewafer add wafers/new.tif --new wafer5
docker compose run --rm whatthewafer list
```

---

## Directory structure

```
WhatTheWafer/
  wafers/              ← place TIF/JPEG images here
  database/            ← feature database (created automatically on first add)
  whatthewafer.tar.gz  ← Docker image archive
  docker-compose.yml
```

`database/` is mounted as a volume — the database persists between container runs
and is directly accessible from the host (sqlite, h5, faiss files).

---

## Diagnostics

```powershell
# Verify the container starts correctly
docker run --gpus all whatthewafer --no-gpu list
# ^ if it doesn't crash — the container works

# Check that the GPU is visible inside the container
docker run --gpus all --entrypoint nvidia-smi whatthewafer
# ^ should print the GPU list

# Check the CUDA version inside the image
docker run --entrypoint python whatthewafer -c "import torch; print(torch.version.cuda)"
```