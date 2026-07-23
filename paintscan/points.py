"""paintscan — shape export to points list.

Turns a photographed pencil/charcoal sketch into an ordered list of boundary
vertices suitable for downstream consumers.

Pipeline
--------
    grayscale
      -> flatten        (divide by heavy Gaussian blur; removes illumination
                         gradient and paper tone)
      -> threshold      (DARKNESS: how committed a line must be to count)
      -> speckle drop   (MINPX)
      -> thinning       (Zhang-Suen; stroke centreline == geometric midline)
      -> spur prune     (PRUNE)
      -> components     (MIN CHAIN)
      -> longest path per component
      -> approxPolyDP(closed=False)

The result is a list of open polylines ("chains") offered to the user as
proposals.  The user deletes, drags and manually joins them in the export
dialog; when a closed ring exists it can be written as a points list.

Nothing here writes to the session JSON — export is a terminal action.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Fixed, not user-facing: measured equivalent across 25/50/100 on real sheets.
FLATTEN_RADIUS = 50

DARKNESS_DEFAULT  = 235
EPS_DEFAULT       = 20     # thousandths of perimeter -> 0.020
MINPX_DEFAULT     = 40
MIN_CHAIN_DEFAULT = 60
PRUNE_DEFAULT     = 8

DARKNESS_MIN,  DARKNESS_MAX  = 180, 254
EPS_MIN,       EPS_MAX       =   2,  60    # thousandths
MINPX_MIN,     MINPX_MAX     =   0, 400
MIN_CHAIN_MIN, MIN_CHAIN_MAX =  10, 400
PRUNE_MIN,     PRUNE_MAX     =   0,  30

# Guards
MIN_RING_VERTICES = 3
MAX_RING_AREA_FRAC = 0.80


# ---------------------------------------------------------------------------
# Config  (repo-local config.json, beside this module)
# ---------------------------------------------------------------------------

# Resolved from __file__ so the file is found regardless of the working
# directory main.py is launched from.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    """Load the config.  Never raises; missing/corrupt -> {}."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(cfg: dict) -> bool:
    """Write the config.  Returns True on success."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        return True
    except Exception as exc:
        print(f"[WARN] Could not write {CONFIG_PATH}: {exc!r}")
        return False


def resolve_points_dir(fallback: Path) -> tuple[Path, bool]:
    """Return (destination_dir, came_from_config).

    Reads ``points_out_dir`` from the config.  Falls back to *fallback* with a
    loud warning when the key is missing or the path is unusable — we never
    silently guess a destination.
    """
    cfg = load_config()
    raw = cfg.get("points_out_dir")
    if not raw:
        print(f"[WARN] No 'points_out_dir' in {CONFIG_PATH} — "
              f"falling back to {fallback}")
        return Path(fallback), False
    try:
        d = Path(raw).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d, True
    except Exception as exc:
        print(f"[WARN] points_out_dir {raw!r} unusable ({exc!r}) — "
              f"falling back to {fallback}")
        return Path(fallback), False


def next_points_path(out_dir: Path) -> Path:
    """Allocate the next free ``from_ps_nnnn.json`` in *out_dir*.

    Scans for existing files rather than keeping a counter, so the numbering
    self-heals if files are moved or deleted by hand.
    """
    out_dir = Path(out_dir)
    highest = 0
    try:
        for p in out_dir.glob("from_ps_*.json"):
            stem = p.stem[len("from_ps_"):]
            if stem.isdigit():
                highest = max(highest, int(stem))
    except Exception:
        pass
    return out_dir / f"from_ps_{highest + 1:04d}.json"


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def flatten(gray: np.ndarray, radius: int = FLATTEN_RADIUS) -> np.ndarray:
    """Remove illumination gradient and paper tone.

    Divides by a heavily blurred copy of itself: the blurred copy *is* the
    illumination field plus paper tone, so dividing it out leaves strokes at
    their true relative darkness on a uniform white ground.
    """
    bg = cv2.GaussianBlur(gray, (0, 0), radius)
    bg = np.maximum(bg, 1)
    flat = (gray.astype(np.float32) / bg.astype(np.float32)) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def prepare(bgr: np.ndarray, radius: int = FLATTEN_RADIUS) -> np.ndarray:
    """BGR image -> flattened grayscale."""
    return flatten(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), radius)


def ink_mask(flat: np.ndarray, darkness: int, min_px: int) -> np.ndarray:
    """Threshold *flat* to a binary ink mask (1 = ink) and drop small specks."""
    ink = (flat < int(darkness)).astype(np.uint8)
    if min_px <= 0:
        return ink
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            keep[lab == i] = 1
    return keep


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------

def _zhang_suen(img: np.ndarray) -> np.ndarray:
    """Vectorised Zhang-Suen thinning.  Dependency-free fallback.

    *img* is uint8 with 1 = foreground.  Returns uint8 with 1 = skeleton.
    """
    s = (img > 0).astype(np.uint8).copy()

    def _pass(s, step):
        p = np.pad(s, 1)
        P2 = p[0:-2, 1:-1]; P3 = p[0:-2, 2:]
        P4 = p[1:-1, 2:];   P5 = p[2:,   2:]
        P6 = p[2:,   1:-1]; P7 = p[2:,   0:-2]
        P8 = p[1:-1, 0:-2]; P9 = p[0:-2, 0:-2]
        seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        A = np.zeros_like(B)
        for i in range(8):
            A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
        if step == 0:
            c1 = (P2 * P4 * P6) == 0
            c2 = (P4 * P6 * P8) == 0
        else:
            c1 = (P2 * P4 * P8) == 0
            c2 = (P2 * P6 * P8) == 0
        rm = (s == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
        s[rm] = 0
        return s, rm.any()

    for _ in range(200):                     # generous cap; converges long before
        s, ch1 = _pass(s, 0)
        s, ch2 = _pass(s, 1)
        if not (ch1 or ch2):
            break
    return s


def thin(ink: np.ndarray) -> np.ndarray:
    """Reduce strokes to single-pixel centrelines (1 = skeleton).

    Uses cv2.ximgproc (opencv-contrib) when present, otherwise the built-in
    Zhang-Suen implementation, so this works on a plain opencv-python install.
    Both paths produce identical output; the fallback is ~75ms at display size.
    """
    src = (ink > 0).astype(np.uint8)
    try:
        return (cv2.ximgproc.thinning(src * 255) > 0).astype(np.uint8)
    except Exception:
        return _zhang_suen(src)


# ---------------------------------------------------------------------------
# Skeleton graph helpers
# ---------------------------------------------------------------------------

_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
       (0, 1), (1, -1), (1, 0), (1, 1))

_DEG_KERNEL = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)


def _neighbours(sk: np.ndarray, y: int, x: int) -> list:
    h, w = sk.shape
    out = []
    for dy, dx in _NB:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and sk[ny, nx]:
            out.append((ny, nx))
    return out


def prune_spurs(sk: np.ndarray, iterations: int) -> np.ndarray:
    """Peel *iterations* pixels off every free end.

    Removes the whiskers that thinning leaves at stroke ends and at crossings,
    without touching anything that is part of a longer run.
    """
    if iterations <= 0:
        return sk
    sk = sk.copy()
    for _ in range(int(iterations)):
        deg  = cv2.filter2D(sk, -1, _DEG_KERNEL)
        ends = (sk > 0) & (deg == 1)
        if not ends.any():
            break
        sk[ends] = 0
    return sk


def _bfs_farthest(sk: np.ndarray, start: tuple) -> tuple:
    """Return (farthest_node, dist_map) by breadth-first search from *start*."""
    dist = {start: 0}
    q = deque([start])
    far = start
    while q:
        cur = q.popleft()
        for nb in _neighbours(sk, cur[0], cur[1]):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
                if dist[nb] > dist[far]:
                    far = nb
    return far, dist


def _longest_path(comp: np.ndarray, seed: tuple) -> list:
    """Longest path through a skeleton component (double-BFS)."""
    a, _    = _bfs_farthest(comp, seed)
    b, dist = _bfs_farthest(comp, a)
    path = [b]
    cur  = b
    guard = comp.sum() + 8
    while cur != a and guard > 0:
        guard -= 1
        nbs = _neighbours(comp, cur[0], cur[1])
        if not nbs:
            break
        cur = min(nbs, key=lambda n: dist.get(n, 1 << 30))
        path.append(cur)
    return path


# ---------------------------------------------------------------------------
# Chain extraction
# ---------------------------------------------------------------------------

def extract_chains(
    ink: np.ndarray,
    min_chain: int = MIN_CHAIN_DEFAULT,
    prune_px: int = PRUNE_DEFAULT,
    eps_thou: int = EPS_DEFAULT,
) -> tuple[list, np.ndarray]:
    """Extract simplified open polylines from an ink mask.

    Returns (chains, skeleton) where *chains* is a list of dicts::

        {"pts": ndarray (N,2) int32 in display px,
         "px":  int   skeleton pixel count,
         "len": float arc length in display px,
         "weight": float mean stroke half-width in display px}

    sorted longest-first.  *skeleton* is returned for display purposes.
    """
    sk = thin(ink)
    sk = prune_spurs(sk, prune_px)

    # Stroke half-width, sampled on the skeleton — used for weight filtering.
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5)

    n, lab = cv2.connectedComponents(sk, 8)
    eps_frac = max(1, int(eps_thou)) / 1000.0

    chains = []
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(ys) < min_chain:
            continue
        comp = (lab == i).astype(np.uint8)
        path = _longest_path(comp, (int(ys[0]), int(xs[0])))
        if len(path) < 2:
            continue
        arr  = np.array([[x, y] for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
        peri = cv2.arcLength(arr, False)
        pts  = cv2.approxPolyDP(arr, eps_frac * peri, False).reshape(-1, 2)
        chains.append({
            "pts":    pts.astype(np.int32),
            "px":     int(len(ys)),
            "len":    float(peri),
            "weight": float(dt[ys, xs].mean()) if len(ys) else 0.0,
        })

    chains.sort(key=lambda c: -c["len"])
    return chains, sk


# ---------------------------------------------------------------------------
# Ring validation and output
# ---------------------------------------------------------------------------

def validate_ring(points: np.ndarray, canvas_w: int, canvas_h: int) -> Optional[str]:
    """Return an error string if *points* is not exportable, else None."""
    if points is None or len(points) < MIN_RING_VERTICES:
        return (f"ring has {0 if points is None else len(points)} vertices; "
                f"at least {MIN_RING_VERTICES} required")
    arr  = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    area = abs(cv2.contourArea(arr))
    frac = area / float(max(1, canvas_w * canvas_h))
    if frac > MAX_RING_AREA_FRAC:
        return (f"ring covers {frac * 100:.0f}% of the canvas — "
                f"selection looks wrong")
    if area <= 0:
        return "ring encloses no area"
    return None


def write_points(path: Path, src_w: int, src_h: int, points) -> Path:
    """Write the points list in the consumer's schema.  Returns *path*."""
    payload = {
        "source": {"w": int(src_w), "h": int(src_h)},
        "points": [[round(float(x), 1), round(float(y), 1)] for x, y in points],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def scale_to_full(points, rx: float, ry: float) -> list:
    """Scale display-resolution points to full-resolution pixels."""
    return [(float(x) * rx, float(y) * ry) for x, y in points]