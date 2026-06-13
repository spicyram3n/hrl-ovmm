# SAM3 inference server.
# SAM3 needs Python >= 3.12, PyTorch >= 2.7 and CUDA >= 12.6, which is
# incompatible with the ROS 2 Humble devcontainer (Ubuntu 22.04, Python 3.10).
# It therefore runs in its own container and is reached over HTTP
# (FastAPI + Uvicorn), same pattern as the OpenMask3D / AnyGrasp servers
# in stretch-compose.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git wget \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# venv to avoid PEP 668 (Ubuntu 24.04 marks system python as externally managed)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch==2.7.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# sam3/__init__.py eagerly imports its training/data-loading code even for
# the inference-only path (build_sam3_image_model -> ... -> sam3.train.data),
# so install the "train" extra to pull in that whole dependency tree
# (hydra-core, omegaconf, fvcore, scipy, scikit-learn/-image, etc.) up front
# instead of discovering each missing import one at a time.
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && pip install --no-cache-dir -e "/opt/sam3[train]"

# Extra runtime deps pulled in by sam3/train/data/* that aren't declared in
# ANY of sam3's pyproject extras (packaging gaps in a freshly-released repo):
#   - einops:      sam3/sam/rope.py
#   - pycocotools, pandas, psutil: sam3/train/data/coco_json_loaders.py and friends
# numpy is pinned <2 to match sam3's pyproject constraint (numpy>=1.26,<2).
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart pillow "numpy<2" \
    opencv-python-headless huggingface_hub einops pycocotools pandas psutil

# Checkpoints are gated on Hugging Face: pass your token at build or run time.
# Either bake the download in (needs HF_TOKEN build arg) or mount a volume
# with the checkpoint and let the app download lazily at first request.
ARG HF_TOKEN=""
ENV HF_TOKEN=$HF_TOKEN

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 5005
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005"]
