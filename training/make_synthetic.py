#!/usr/bin/env python3
"""
make_synthetic.py

Synthetic YOLO dataset of exit signs: signs drawn by exit_inspection.sign_art (ISO 7010 E001/E002, arrow panels,
text bands, lit box frames), warped in perspective and pasted onto background photos, plus unlabeled distractors.

  <out>/images/{train,val}/NNNNNN.jpg   1280x720
  <out>/labels/{train,val}/NNNNNN.txt   class x_center y_center width height (normalized)
  <out>/data.yaml                       ultralytics dataset file (classes of exit_inspection.sign_art.CLASSES)

Backgrounds: every image of the --backgrounds folders (searched recursively); a folder with a clean_backgrounds.txt
(sim/harvest_renders.py output) contributes only the frames listed there, where no part of any sign is in view, so no
unlabeled sign gets in.
Sign sizes: 8-100 px tall at 1280x720 (12-150 px at 1080p); view angles up to 65 deg sideways, 30 deg from below.
Distractors: other green safety signs (first aid, eyewash, plain panels), red fire signs, green lights, screens.

Usage (.venv-yolo python):
  python training/make_synthetic.py ~/RaNAV/datasets/exit_sign/synthetic --n-train 8000 --n-val 800 \
      --backgrounds ~/RaNAV/datasets/voc2007_dl ~/RaNAV/datasets/coco128_dl ~/RaNAV/datasets/exit_sign/sim_backgrounds:10
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from exit_inspection.sign_art import ARROWS, CLASSES, FONTS, GREEN, SignSpec, render  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT_W, OUT_H = 1280, 720
TEXTS = [None, None, None, "USCITA DI EMERGENZA", "USCITA", "EXIT", "EMERGENCY EXIT", "SALIDA", "SORTIE",
         "NOTAUSGANG", "USCITA DI SICUREZZA"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def list_backgrounds(folders):
    """Background paths; a folder given as DIR:N counts N times (to weigh a small folder against a big one)."""
    out = []
    for spec in folders:
        folder, _, repeat = spec.partition(":")
        folder = os.path.expanduser(folder)
        found = []
        clean = os.path.join(folder, "clean_backgrounds.txt")
        if os.path.exists(clean):
            with open(clean) as f:
                found = [os.path.join(folder, line.strip()) for line in f if line.strip()]
        else:
            for root, _, files in os.walk(folder):
                found += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXT)]
        print(f"{len(found)} backgrounds in {folder} x {int(repeat or 1)}")
        out += found * int(repeat or 1)
    return out


def background(path, rng):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    target = OUT_W / OUT_H
    if w / h > target:
        cw, ch = int(h * target), h
    else:
        cw, ch = w, int(w / target)
    x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    img = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    return cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def random_spec(rng):
    arrow = ARROWS[rng.choice(len(ARROWS), p=_ARROW_P)]
    text = TEXTS[rng.integers(len(TEXTS))]
    return SignSpec(pictogram=str(rng.choice(["E001", "E002"])), arrow=arrow, arrow_first=bool(rng.random() < 0.5),
                    text=text, font=int(rng.integers(len(FONTS))))


# None 35 %, down 25 %, the six side directions share 40 %
_ARROW_P = np.array([0.35, 0.25] + [0.4 / 7] * 7)
_ARROW_P /= _ARROW_P.sum()


def sign_rgba(spec, rng, px=128):
    """Sign image (BGR) and alpha, with an optional lit-box frame around it."""
    img = cv2.cvtColor(np.array(render(spec, px)), cv2.COLOR_RGB2BGR)
    if rng.random() < 0.5:  # frame of a lit sign box: white, grey or aluminium
        t = int(px * rng.uniform(0.04, 0.12))
        color = [int(c) for c in rng.choice([[245, 245, 245], [200, 200, 200], [170, 175, 180], [60, 60, 60]])]
        img = cv2.copyMakeBorder(img, t, t, t, t, cv2.BORDER_CONSTANT, value=color)
    return img, np.full(img.shape[:2], 255, np.uint8)


def distractor_rgba(rng, px=128):
    """Unlabeled look-alikes: other safety signs, plain panels, lights, screens."""
    kind = rng.choice(["first_aid", "eyewash", "green_plain", "green_text", "fire", "light", "screen"])
    if kind == "light":
        s = int(px * rng.uniform(0.1, 0.4))
        img = np.zeros((s * 3, s * 3, 3), np.uint8)
        cv2.circle(img, (int(1.5 * s), int(1.5 * s)), s // 2 + 1, (90, 255, 120), -1)
        img = cv2.GaussianBlur(img, (0, 0), s / 2) + img
        alpha = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8) * 255
        return img, alpha
    if kind == "screen":
        w, h = int(px * rng.uniform(1.2, 2.5)), int(px * rng.uniform(0.7, 1.4))
        img = np.full((h, w, 3), rng.integers(0, 60, 3), np.uint8)
        img[int(h * 0.1):int(h * 0.9), int(w * 0.05):int(w * 0.95)] = rng.integers(0, 255, 3)
        return img, np.full((h, w), 255, np.uint8)
    green = tuple(int(c) for c in GREEN[::-1])  # BGR
    bg = (40, 40, 200) if kind == "fire" else green
    w = int(px * (1.0 if kind in ("first_aid", "eyewash", "fire") else rng.uniform(1.0, 3.0)))
    img = Image.new("RGB", (w, px), bg[::-1])
    d = ImageDraw.Draw(img)
    c = px // 2
    if kind == "first_aid":
        a, b = px * 0.12, px * 0.36
        d.rectangle([w / 2 - a, c - b, w / 2 + a, c + b], fill="white")
        d.rectangle([w / 2 - b, c - a, w / 2 + b, c + a], fill="white")
    elif kind == "eyewash":
        d.ellipse([w / 2 - px * 0.3, c - px * 0.15, w / 2 + px * 0.3, c + px * 0.15], outline="white",
                  width=px // 14)
        d.ellipse([w / 2 - px * 0.08, c - px * 0.08, w / 2 + px * 0.08, c + px * 0.08], fill="white")
        for k in range(3):
            d.line([w / 2 - px * 0.2 + k * px * 0.2, c + px * 0.22, w / 2 - px * 0.2 + k * px * 0.2, c + px * 0.4],
                   fill="white", width=px // 20)
    elif kind == "green_text":
        f = ImageFont.truetype(FONTS[0], int(px * 0.3))
        d.text((w / 2, c), str(rng.choice(["PRONTO SOCCORSO", "FIRST AID", "PUNTO DI RACCOLTA", "INFO"])),
               font=f, fill="white", anchor="mm")
    elif kind == "fire":
        d.polygon([(w * 0.5, px * 0.15), (w * 0.72, px * 0.6), (w * 0.5, px * 0.85), (w * 0.28, px * 0.6)],
                  fill="white")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img, np.full(img.shape[:2], 255, np.uint8)


def warp_quad(w, h, height_px, rng, max_yaw=65.0):
    """Destination quad (4x2, image px) of a w x h planar sign seen in perspective, about height_px tall."""
    yaw = math.radians(rng.uniform(-max_yaw, max_yaw))
    pitch = math.radians(rng.uniform(-30, 10))  # negative: seen from below
    roll = math.radians(rng.normal(0, 2.5))
    aspect = w / h
    pts = np.array([[-aspect / 2, -0.5, 0], [aspect / 2, -0.5, 0], [aspect / 2, 0.5, 0], [-aspect / 2, 0.5, 0]])
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), \
        math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    p = pts @ (Rz @ Rx @ Ry).T
    dist = rng.uniform(4.0, 12.0)  # in sign heights: perspective strength
    p[:, 2] += dist
    uv = p[:, :2] / p[:, 2:3]
    scale = height_px / (uv[:, 1].max() - uv[:, 1].min())
    return uv * scale


def photometric(img, rng):
    img = img.astype(np.float32)
    img *= rng.uniform(0.55, 1.35)  # brightness
    if rng.random() < 0.25:  # overexposed lit sign: whites clip, the green gets lighter
        img = img * rng.uniform(1.15, 1.6) + rng.uniform(0, 35)
    if rng.random() < 0.5:  # hue / white balance shift
        img *= rng.uniform(0.9, 1.1, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def paste(bg, sign, alpha, quad, rng, glow):
    h, w = sign.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    warped = cv2.warpPerspective(sign, M, (OUT_W, OUT_H), flags=cv2.INTER_AREA)
    a = cv2.warpPerspective(alpha, M, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    out = bg.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a
    if glow:
        halo = cv2.GaussianBlur(warped.astype(np.float32) * a, (0, 0), max(1.0, 0.15 * (np.ptp(quad[:, 1]))))
        out += halo * rng.uniform(0.2, 0.6)
    return np.clip(out, 0, 255).astype(np.uint8)


def place(quad, rng, occupied):
    """Shift the quad to a random spot (upper part more likely) not overlapping earlier ones; None if none found."""
    qw, qh = np.ptp(quad[:, 0]), np.ptp(quad[:, 1])
    for _ in range(20):
        x = rng.uniform(0, OUT_W - qw)
        y = OUT_H * min(rng.beta(1.3, 2.5), 0.999) * (1 - qh / OUT_H)
        q = quad - quad.min(axis=0) + [x, y]
        box = (*q.min(axis=0), *q.max(axis=0))
        if all(box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3] for o in occupied):
            return q, box
    return None, None


def image_noise(img, rng):
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, M, (k, k)))
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.6))
    sigma = rng.uniform(0, 7)
    if sigma > 0.5:
        img = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    q = int(rng.integers(40, 96))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def make_image(bg_path, rng):
    img = background(bg_path, rng)
    labels, occupied = [], []
    n_signs = int(rng.choice([0, 1, 1, 1, 2, 2, 3, 4]))
    n_distractors = int(rng.choice([0, 0, 1, 1, 2]))
    items = [("sign", None)] * n_signs + [("distractor", None)] * n_distractors
    rng.shuffle(items)
    for kind, _ in items:
        if kind == "sign":
            spec = random_spec(rng)
            sign, alpha = sign_rgba(spec, rng)
        else:
            sign, alpha = distractor_rgba(rng)
        height = math.exp(rng.uniform(math.log(8), math.log(100)))
        quad = warp_quad(sign.shape[1], sign.shape[0], height, rng)
        quad, box = place(quad, rng, occupied)
        if quad is None:
            continue
        occupied.append(box)
        img = paste(img, photometric(sign, rng), alpha, quad, rng, glow=kind == "sign" and rng.random() < 0.4)
        if kind == "sign":
            x1, y1, x2, y2 = box
            labels.append((CLASSES.index(spec.cls), (x1 + x2) / 2 / OUT_W, (y1 + y2) / 2 / OUT_H, (x2 - x1) / OUT_W,
                           (y2 - y1) / OUT_H))
    return image_noise(img, rng), labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--backgrounds", nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bgs = list_backgrounds(args.backgrounds)
    if not bgs:
        sys.exit("no background images found")
    print(f"{len(bgs)} backgrounds")
    rng = np.random.default_rng(args.seed)
    out = os.path.expanduser(args.out)
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
        n_labels = 0
        for i in range(n):
            img, labels = make_image(bgs[rng.integers(len(bgs))], rng)
            name = f"{i:06d}"
            cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(out, "labels", split, name + ".txt"), "w") as f:
                f.writelines(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels)
            n_labels += len(labels)
            if i % 500 == 0:
                print(f"{split} {i}/{n}", flush=True)
        print(f"{split}: {n} images, {n_labels} signs")
    with open(os.path.join(out, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": out, "train": "images/train", "val": "images/val",
                        "names": {i: c for i, c in enumerate(CLASSES)}}, f, sort_keys=False)
    print("wrote", os.path.join(out, "data.yaml"))


if __name__ == "__main__":
    main()
