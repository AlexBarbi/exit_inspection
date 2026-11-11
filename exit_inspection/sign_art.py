"""Exit sign images: an ISO 7010 E001/E002 pictogram, an optional arrow panel, an optional text band.

Used by the Gazebo world generator (textures) and the synthetic training data. The pictograms are the Wikimedia
Commons renders of ISO_7010_E001.svg / ISO_7010_E002.svg (public domain, see training/assets/iso7010/README.md); the
arrows and the text are drawn here.

Detector classes: no arrow -> exit_plain, arrow down -> exit_arrow_down, any other arrow -> exit_arrow_side.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "iso7010")
CLASSES = ["exit_plain", "exit_arrow_down", "exit_arrow_side"]
ARROWS = [None, "down", "left", "right", "up", "up_left", "up_right", "down_left", "down_right"]
GREEN = (36, 126, 81)  # ISO 7010 safety green as in the Commons renders
WHITE = (255, 255, 255)
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"]
_ARROW_ANGLE = {"right": 0, "up_right": 45, "up": 90, "up_left": 135, "left": 180, "down_left": 225, "down": 270,
                "down_right": 315}


def class_of(arrow: Optional[str]) -> str:
    if arrow is None:
        return "exit_plain"
    return "exit_arrow_down" if arrow == "down" else "exit_arrow_side"


@dataclass
class SignSpec:
    pictogram: str = "E001"  # E001 (runs left) or E002 (runs right)
    arrow: Optional[str] = None  # one of ARROWS
    arrow_first: bool = False  # arrow panel left of the pictogram
    text: Optional[str] = None  # band under the panels
    font: int = 0  # index in FONTS

    @property
    def cls(self) -> str:
        return class_of(self.arrow)


def pictogram(name: str, size: int) -> Image.Image:
    img = Image.open(os.path.join(ASSETS, f"ISO_7010_{name}.png")).convert("RGB")
    # the renders have a 2.5 % white margin around the green square: crop it
    w = img.width
    m = int(round(0.025 * w))
    return img.crop((m, m, w - m, w - m)).resize((size, size), Image.LANCZOS)


def arrow_panel(direction: str, size: int) -> Image.Image:
    """White arrow on a green square, pointing `direction`."""
    s = 4 * size  # drawn large, then downscaled for smooth edges
    img = Image.new("RGB", (s, s), GREEN)
    d = ImageDraw.Draw(img)
    # arrow pointing right, centered: shaft + head, in units of s
    pts = np.array([[-0.36, -0.09], [0.06, -0.09], [0.06, -0.27], [0.38, 0.0], [0.06, 0.27], [0.06, 0.09],
                    [-0.36, 0.09]])
    a = np.radians(_ARROW_ANGLE[direction])
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # image y points down
    pts = pts @ rot.T * s + s / 2
    d.polygon([tuple(p) for p in pts], fill=WHITE)
    return img.resize((size, size), Image.LANCZOS)


def text_band(text: str, width: int, height: int, font: int = 0) -> Image.Image:
    img = Image.new("RGB", (width, height), GREEN)
    d = ImageDraw.Draw(img)
    size = int(height * 0.7)
    f = ImageFont.truetype(FONTS[font % len(FONTS)], size)
    while size > 6 and d.textlength(text, font=f) > 0.92 * width:
        size -= 1
        f = ImageFont.truetype(FONTS[font % len(FONTS)], size)
    tw = d.textlength(text, font=f)
    d.text(((width - tw) / 2, height / 2), text, font=f, fill=WHITE, anchor="lm")
    return img


def render(spec: SignSpec, height: int = 256) -> Image.Image:
    """Sign image; the pictogram is height x height (without the text band)."""
    tiles = [pictogram(spec.pictogram, height)]
    if spec.arrow is not None:
        arrow = arrow_panel(spec.arrow, height)
        tiles = [arrow, tiles[0]] if spec.arrow_first else [tiles[0], arrow]
    gap = max(1, height // 40)  # thin white line between the panels, as on printed signs
    width = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    band = int(0.32 * height) if spec.text else 0
    img = Image.new("RGB", (width, height + band + (gap if band else 0)), WHITE)
    x = 0
    for t in tiles:
        img.paste(t, (x, 0))
        x += t.width + gap
    if band:
        img.paste(text_band(spec.text, width, band, spec.font), (0, height + gap))
    return img
