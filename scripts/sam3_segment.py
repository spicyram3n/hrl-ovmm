#!/usr/bin/env python3
"""
Minimal host-side client for the SAM3 docker server (docker/sam3).

1. Start the server (separate terminal):
       bash docker/sam3/run_sam3.sh
   It serves FastAPI/Uvicorn on http://localhost:5005

2. Run this script (needs: pip install requests numpy opencv-python):
       python3 scripts/sam3_segment.py <image_path> "<prompt>" [conf]

   Example:
       python3 scripts/sam3_segment.py \
           /home/comrade/homeobjects-3K/images/train/living_room_1001.jpg \
           "sofa" 0.5

Saves an overlay of the predicted masks/boxes to sam3_output.jpg.
"""

import sys

import cv2
import numpy as np

from core.perception.detection.sam3_client import Sam3Client


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image_path> <prompt> [conf=0.5]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    img = cv2.imread(image_path)
    det = Sam3Client().detect(img, prompt, conf=conf)
    n, h, w = det.masks.shape

    # SAM3 runs at a fixed resolution, so resize the source image to match
    # before overlaying masks/boxes.
    img = cv2.resize(img, (w, h))

    rng = np.random.default_rng(0)
    for i in range(n):
        color = rng.integers(0, 255, 3).tolist()
        img[det.masks[i]] = (0.5 * img[det.masks[i]] + 0.5 * np.array(color)).astype(np.uint8)
        x1, y1, x2, y2 = det.boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{prompt} {det.scores[i]:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = "sam3_output.jpg"
    cv2.imwrite(out_path, img)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
