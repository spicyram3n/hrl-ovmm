"""
SAM3 inference server (FastAPI + Uvicorn).

Wraps the official SAM3 image-inference pattern
(https://github.com/facebookresearch/sam3) over HTTP so it can run in its
own CUDA 12.6 / Python 3.12 container, separate from the ROS 2 Humble
(Python 3.10) workspace:

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt="...")
    # output = {"masks": (N,H,W) bool, "boxes": (N,4) xyxy, "scores": (N,)}

The model is loaded once and reused for every request — this server is
meant to be called once per video frame from a ROS 2 node (see
core/perception/detection/sam3_client.py), so per-request overhead matters.

Endpoint:
    POST /sam3/predict
        files:  image  (jpg/png)
        params: prompt (text concept, e.g. "red cup")
                conf   (optional float threshold, default 0.5)

Response: raw bytes = masks.tobytes() + boxes.tobytes() + scores.tobytes()
Header 'X-Sam3-Meta': json {"num_instances": N, "height": H, "width": W}
"""

import io
import json
import os
import threading

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI()

# Sam3Processor resizes every image to a fixed resolution (1008x1008 by
# default), so cuDNN can pick fixed conv algorithms once and reuse them on
# every frame instead of re-benchmarking per call.
torch.backends.cudnn.benchmark = True

MODEL = None
PROCESSOR = None
_MODEL_LOCK = threading.Lock()


def get_model():
    """Lazy-load on first request; thread-safe against concurrent requests."""
    global MODEL, PROCESSOR
    if MODEL is None:
        with _MODEL_LOCK:
            if MODEL is None:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor

                token = os.environ.get("HF_TOKEN") or None
                if token:
                    from huggingface_hub import login
                    login(token=token)

                model = build_sam3_image_model()  # downloads facebook/sam3 checkpoint
                PROCESSOR = Sam3Processor(model)
                MODEL = model
    return MODEL, PROCESSOR


@app.post("/sam3/predict")
async def predict(image: UploadFile = File(...), prompt: str = "", conf: float = 0.5):
    if not prompt:
        return Response(content=json.dumps({"error": "no prompt provided"}), status_code=400)

    img = Image.open(io.BytesIO(await image.read())).convert("RGB")

    _, processor = get_model()
    # SAM3's checkpoint mixes bf16 and fp32 weights across submodules.
    # autocast casts both operands of every matmul to bf16 at call time,
    # so the stored weight dtypes never need to match each other.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(img)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]    # (N, 1, H, W) bool
    boxes = output["boxes"]    # (N, 4) xyxy
    scores = output["scores"]  # (N,)

    # .float() first: autocast may return bf16 tensors, which .numpy()
    # cannot convert directly.
    masks = masks.float().cpu().numpy().astype(bool)
    boxes = boxes.float().cpu().numpy().astype(np.float32)
    scores = scores.float().cpu().numpy().astype(np.float32)

    # Drop the singleton mask-channel dim: (N, 1, H, W) -> (N, H, W).
    # This is NOT a batch dim - N is the per-detection axis, same as boxes/scores.
    if masks.ndim == 4:
        masks = masks[:, 0]

    keep = scores >= conf
    masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
    n, h, w = masks.shape

    print(f"[sam3] '{prompt}': {n} instance(s)")
    body = masks.tobytes() + boxes.tobytes() + scores.tobytes()
    meta = json.dumps({"num_instances": n, "height": h, "width": w})
    return Response(content=body, media_type="application/octet-stream", headers={"X-Sam3-Meta": meta})


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
