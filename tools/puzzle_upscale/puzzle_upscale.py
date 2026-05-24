# puzzle_upscale.py

from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse

# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Input image"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output PNG"
)

parser.add_argument(
    "--height",
    type=int,
    default=8400,
    help="Target height in pixels"
)

parser.add_argument(
    "--sat",
    type=float,
    default=1.18,
    help="Saturation boost"
)

parser.add_argument(
    "--contrast",
    type=float,
    default=1.10,
    help="Contrast boost"
)

parser.add_argument(
    "--sharp",
    type=float,
    default=2.0,
    help="Sharpness boost"
)

args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output

TARGET_HEIGHT = args.height

SATURATION = args.sat
CONTRAST = args.contrast
SHARPNESS = args.sharp

# =========================================================
# LOAD
# =========================================================

img = Image.open(INPUT_FILE).convert("RGB")

w, h = img.size
aspect = w / h

target_width = int(TARGET_HEIGHT * aspect)

print(f"Original : {w} x {h}")
print(f"Upscaled : {target_width} x {TARGET_HEIGHT}")

# =========================================================
# HIGH QUALITY UPSCALE
# =========================================================

img = img.resize(
    (target_width, TARGET_HEIGHT),
    Image.Resampling.LANCZOS
)

# =========================================================
# REMOVE JPEG ARTIFACTS
# =========================================================

img = img.filter(ImageFilter.MedianFilter(size=3))

# =========================================================
# COLOR / CONTRAST
# =========================================================

img = ImageEnhance.Color(img).enhance(SATURATION)
img = ImageEnhance.Contrast(img).enhance(CONTRAST)

# =========================================================
# EDGE SHARPENING
# =========================================================

img = ImageEnhance.Sharpness(img).enhance(SHARPNESS)

img = img.filter(
    ImageFilter.UnsharpMask(
        radius=1.8,
        percent=180,
        threshold=3
    )
)

# =========================================================
# OPENCV EDGE BOOST
# =========================================================

np_img = np.array(img)

edges = cv2.Canny(np_img, 80, 180)

edges = cv2.dilate(
    edges,
    np.ones((2, 2), np.uint8),
    iterations=1
)

edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

boosted = cv2.addWeighted(
    np_img,
    1.0,
    edges_rgb,
    0.08,
    0
)

# =========================================================
# SAVE
# =========================================================

final = Image.fromarray(boosted)

Path("output").mkdir(exist_ok=True)

final.save(
    OUTPUT_FILE,
    format="PNG",
    compress_level=1
)

print()
print("DONE")
print(f"Saved to: {OUTPUT_FILE}")