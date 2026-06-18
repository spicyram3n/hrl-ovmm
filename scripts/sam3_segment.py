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

from core.perception.detection.sam3_client import Sam3Client, draw_detections


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image_path> <prompt> [conf=0.5]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    img = cv2.imread(image_path)
    det = Sam3Client().detect(img, prompt, conf=conf)

    # SAM3 runs at a fixed resolution, so resize the source image to match
    # before overlaying masks/boxes.
    _, h, w = det.masks.shape
    img = cv2.resize(img, (w, h))

    out_path = "sam3_output.jpg"
    cv2.imwrite(out_path, draw_detections(img, det, prompt))
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
