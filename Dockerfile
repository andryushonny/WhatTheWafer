FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# OpenCV runtime libs (headless build needs libGL + libGLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — torch/torchvision already provided by the base image
COPY requirements.txt .
RUN pip install --no-cache-dir \
        "kornia>=0.8.0" \
        "opencv-python-headless>=4.8.0" \
        "tifffile>=2023.1.0" \
        "scikit-image>=0.20.0" \
        "h5py>=3.8.0" \
        "faiss-cpu>=1.7.0"

# Application code + bundled model weights (models/checkpoints/)
COPY . .

ENTRYPOINT ["python", "wafer_id.py"]
