#!/usr/bin/env python3
"""
evaluate.py

Recall of the exit sign detector on labeled frames (sim/harvest_renders.py output: images/, labels/, meta.csv), by
distance, box height and view angle, plus false positives per frame. A sign counts as found when a detection of any
class overlaps its box (IoU >= --iou); class accuracy is reported among the found ones.

Gate of the spec: recall >= 0.9 on the sim renders up to 10 m.

Usage (.venv-yolo python):
  python training/evaluate.py ~/RaNAV/models/exit_sign/exit_sign_736x1280.onnx ~/RaNAV/datasets/exit_sign/sim_renders
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from exit_inspection.sign_art import CLASSES  # noqa: E402


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load_labels(path, w, h):
    out = []
    if os.path.exists(path):
        for line in open(path):
            c, x, y, bw, bh = map(float, line.split())
            out.append((int(c), [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]))
    return out


def binned(rows, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]
        if sel:
            out.append((f"{lo:g}-{hi:g}", len(sel), sum(r["found"] for r in sel) / len(sel)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("frames")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--imgsz", default="736,1280")
    ap.add_argument("--max-distance", type=float, default=10.0)
    ap.add_argument("--gate", type=float, default=0.9)
    args = ap.parse_args()

    imgsz = [int(v) for v in args.imgsz.split(",")]
    model = YOLO(args.weights, task="detect")
    meta = defaultdict(list)
    with open(os.path.join(args.frames, "meta.csv")) as f:
        for r in csv.DictReader(f):
            meta[r["frame"]].append(r)
    rows, fps, n_frames = [], 0, 0
    images = sorted(glob.glob(os.path.join(args.frames, "images", "*.jpg")))
    step = 1 if args.weights.endswith(".onnx") else 16  # the deployed ONNX has a static batch of 1
    for i in range(0, len(images), step):
        batch = images[i:i + step]
        for path, res in zip(batch, model.predict(batch, imgsz=imgsz, conf=args.conf, verbose=False)):
            n_frames += 1
            h, w = res.orig_shape
            name = os.path.splitext(os.path.basename(path))[0]
            gts = load_labels(os.path.join(args.frames, "labels", name + ".txt"), w, h)
            dets = [(int(c), b.tolist()) for c, b in zip(res.boxes.cls.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
            used = set()
            for k, (gc, gb) in enumerate(gts):
                best, bj = 0.0, None
                for j, (dc, db) in enumerate(dets):
                    if j in used:
                        continue
                    v = iou(gb, db)
                    if v > best:
                        best, bj = v, j
                found = best >= args.iou
                if found:
                    used.add(bj)
                m = meta[name][k] if k < len(meta[name]) else {}
                rows.append({"found": found, "cls_ok": found and dets[bj][0] == gc, "cls": CLASSES[gc],
                             "distance": float(m.get("distance", 0)), "angle": float(m.get("view_angle_deg", 0)),
                             "height": gb[3] - gb[1]})
            fps += len(dets) - len(used)

    if not rows:
        sys.exit("no labeled signs")
    found = [r for r in rows if r["found"]]
    print(f"{n_frames} frames, {len(rows)} signs: recall {len(found) / len(rows):.3f}, class accuracy among found "
          f"{sum(r['cls_ok'] for r in found) / max(1, len(found)):.3f}, false positives {fps} "
          f"({fps / n_frames:.3f} per frame)")
    for title, key, edges in (("distance [m]", "distance", [0, 5, 7.5, 10, 12.5, 15]),
                              ("box height [px]", "height", [0, 15, 25, 40, 70, 2000]),
                              ("view angle [deg]", "angle", [0, 30, 45, 60, 70, 81])):
        print(f"recall by {title}: " + ", ".join(f"{b} ({n}): {r:.2f}" for b, n, r in binned(rows, key, edges)))
    for c in CLASSES:
        sel = [r for r in rows if r["cls"] == c]
        if sel:
            print(f"  {c}: {len(sel)} signs, recall {sum(r['found'] for r in sel) / len(sel):.2f}, class ok "
                  f"{sum(r['cls_ok'] for r in sel) / len(sel):.2f}")
    near = [r for r in rows if r["distance"] <= args.max_distance]
    recall = sum(r["found"] for r in near) / max(1, len(near))
    ok = recall >= args.gate
    print(f"GATE recall up to {args.max_distance:g} m: {recall:.3f} over {len(near)} signs -> "
          f"{'PASS' if ok else 'FAIL'} (>= {args.gate})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
