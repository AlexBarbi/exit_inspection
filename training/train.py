#!/usr/bin/env python3
"""
train.py

Fine-tunes YOLO26n (the COCO weights of opti_pessi_perception) on the synthetic exit sign dataset and exports the
best weights to ONNX at a fixed 736x1280 input, the layout Yolo26RGBD (YOLOs-CPP) runs: raw head, NMS in the node
(nms unset, like weights/yolo26n.onnx), static shape (a dynamic one would make YOLOs-CPP fall back to 640x640).

Augmentations: horizontal flips on (E001 <-> E002 and left <-> right arrows keep their class), vertical flips off
(an arrow down must stay down), small hue jitter (the safety green is the cue).

Usage (.venv-yolo python):
  python training/train.py --data ~/RaNAV/datasets/exit_sign/synthetic/data.yaml --epochs 60 \
      --export ~/RaNAV/models/exit_sign/exit_sign_736x1280.onnx
"""

import argparse
import os
import shutil

from ultralytics import YOLO

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(os.path.dirname(PKG))
BASE = os.path.join(WORKSPACE, "src", "legged_control", "opti_pessi_perception", "weights", "yolo26n.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=BASE)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--project", default=os.path.expanduser("~/RaNAV/datasets/exit_sign/runs"))
    ap.add_argument("--name", default="exit_sign")
    ap.add_argument("--export", default=os.path.expanduser("~/RaNAV/models/exit_sign/exit_sign_736x1280.onnx"))
    ap.add_argument("--export-only", help="skip training, export these weights")
    args = ap.parse_args()

    if args.export_only:
        best = args.export_only
    else:
        model = YOLO(args.weights)
        model.train(data=os.path.expanduser(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, project=args.project, name=args.name, exist_ok=False, patience=20,
                    fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, degrees=0.0, mosaic=1.0,
                    close_mosaic=10, scale=0.5, cache=False, plots=True)
        best = os.path.join(str(model.trainer.save_dir), "weights", "best.pt")
    onnx = YOLO(best).export(format="onnx", imgsz=(736, 1280), dynamic=False, simplify=True)
    os.makedirs(os.path.dirname(args.export), exist_ok=True)
    shutil.copy(onnx, args.export)
    print(f"best weights {best}\nONNX {args.export}")


if __name__ == "__main__":
    main()
