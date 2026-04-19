from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from interact import edit_quad, edit_edgemap
from utils import (
    draw_quad,
    load_image,
    order_corners,
    resize_for_preview,
    save_jpg,
    scale_points,
)


@dataclass
class ScanConfig:
    downscale_max_dim: int = 1600
    blur_ksize: int = 5
    canny_lo: int = 50
    canny_hi: int = 150
    contour_min_area_ratio: float = 0.03
    approx_epsilon_ratio: float = 0.02
    jpg_quality: int = 95
    trim_px: int = 2
    debug: bool = False
    interactive: bool = True
    edgemap: bool = False


@dataclass
class CandidateQuad:
    corners: np.ndarray      # (4, 2) float32 in preview coordinates
    area: float
    score: float
    method: str = "contour"


@dataclass
class ProcessResult:
    source: Path
    output_master: Path | None
    confidence: float | None
    method: str | None
    accepted: bool
    used_interactive: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Basic image helpers
# ---------------------------------------------------------------------------

def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_gray(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def edge_map(gray_blur: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return cv2.Canny(gray_blur, lo, hi)


def quad_area(corners: np.ndarray) -> float:
    pts = corners.astype(np.float32)
    return abs(cv2.contourArea(pts.reshape((-1, 1, 2))))


def is_convex_quad(corners: np.ndarray) -> bool:
    pts = corners.astype(np.float32).reshape((-1, 1, 2))
    return bool(cv2.isContourConvex(pts))


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(corners: np.ndarray, image_shape: tuple[int, int, int]) -> float:
    """
    Score a quadrilateral candidate on three criteria:
      - area ratio (larger canvas → higher score)
      - top/bottom width balance
      - left/right height balance

    Returns 0.0 for non-convex quads and for any quad whose corner lands
    within 3 % of the image boundary — that almost always means the detector
    latched onto the floor, wall or easel rather than the canvas.
    """
    h, w = image_shape[:2]
    image_area = float(h * w)
    area = quad_area(corners)
    area_ratio = area / image_area if image_area > 0 else 0.0

    if not is_convex_quad(corners):
        return 0.0

    boundary_margin = 0.03
    for (cx, cy) in corners:
        if cx < w * boundary_margin or cx > w * (1.0 - boundary_margin):
            return 0.0
        if cy < h * boundary_margin or cy > h * (1.0 - boundary_margin):
            return 0.0

    rect = order_corners(corners)
    tl, tr, br, bl = rect

    top_w    = np.linalg.norm(tr - tl)
    bot_w    = np.linalg.norm(br - bl)
    left_h   = np.linalg.norm(bl - tl)
    right_h  = np.linalg.norm(br - tr)

    width_balance  = min(top_w,  bot_w)  / max(top_w,  bot_w)  if max(top_w,  bot_w)  > 0 else 0.0
    height_balance = min(left_h, right_h) / max(left_h, right_h) if max(left_h, right_h) > 0 else 0.0

    score = (
        0.65 * min(max(area_ratio, 0.0), 1.0)
        + 0.175 * width_balance
        + 0.175 * height_balance
    )
    return float(score)


# ---------------------------------------------------------------------------
# Detection method 1: contour-based
# ---------------------------------------------------------------------------

def detect_contour_quad(image_small: np.ndarray, cfg: ScanConfig) -> CandidateQuad | None:
    """
    Find the canvas by looking for large closed contours in the Canny edge map.

    Because paintings are edge-rich internally, we run the search twice —
    once on the raw edge map and once on a morphologically-closed version
    that bridges gaps in the canvas outline — and keep the best result.

    For each large contour we try (in order):
      1. approxPolyDP directly
      2. approxPolyDP on the convex hull (with a sweep of epsilon values)
      3. minAreaRect of the hull as a last resort
    """
    gray      = to_gray(image_small)
    gray_blur = blur_gray(gray, cfg.blur_ksize)
    edges     = edge_map(gray_blur, cfg.canny_lo, cfg.canny_hi)

    # Collect contours from both raw and morphologically-closed edges
    contours_raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours_closed, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours_raw) + list(contours_closed)

    if not contours:
        return None

    h, w   = image_small.shape[:2]
    min_area = cfg.contour_min_area_ratio * (h * w)
    best: CandidateQuad | None = None

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < min_area:
            break                   # sorted descending — no point continuing

        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.approx_epsilon_ratio * peri, True)

        if len(approx) != 4:
            # Fallback 1: hull + epsilon sweep
            hull      = cv2.convexHull(c)
            hull_peri = cv2.arcLength(hull, True)
            approx    = None
            for eps in [0.02, 0.04, 0.06, 0.08]:
                candidate_approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
                if len(candidate_approx) == 4:
                    approx = candidate_approx
                    break

            if approx is None or len(approx) != 4:
                # Fallback 2: minimum-area bounding rectangle
                rect = cv2.minAreaRect(c)
                box  = cv2.boxPoints(rect)
                approx = box.reshape(4, 1, 2).astype(np.int32)

        corners = order_corners(approx.reshape(4, 2).astype(np.float32))
        score   = score_candidate(corners, image_small.shape)
        if score <= 0:
            continue

        cand = CandidateQuad(corners=corners, area=area, score=score, method="contour")
        if best is None or cand.score > best.score:
            best = cand

    return best


# ---------------------------------------------------------------------------
# Detection method 2: Hough line-based
# ---------------------------------------------------------------------------

def detect_line_quad(image_small: np.ndarray, cfg: ScanConfig) -> CandidateQuad | None:
    """
    Detect canvas edges as straight lines via HoughLinesP and intersect them.

    Improvements over the original:
    - Lines are required to span at least 30 % of the relevant image dimension
      (width for verticals, height for horizontals).  Short lines inside the
      painting are therefore filtered out.
    - Clusters are weighted by total line length so longer, more reliable lines
      dominate the representative chosen for intersection.
    - Boundary and convexity checks are applied before returning.
    """
    gray      = to_gray(image_small)
    gray_blur = blur_gray(gray, cfg.blur_ksize)
    edges     = edge_map(gray_blur, cfg.canny_lo, cfg.canny_hi)
    h, w      = image_small.shape[:2]

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(h, w) * 0.20,   # must span ≥20 % of longest dimension
        maxLineGap=30,
    )
    if lines is None:
        return None

    lines = lines.reshape(-1, 4)

    horizontals: list[dict] = []
    verticals:   list[dict] = []

    for x1, y1, x2, y2 in lines:
        dx     = x2 - x1
        dy     = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 40:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180.0 - angle

        if angle < 15:                              # near-horizontal
            # Require the line to span ≥30 % of image width
            x_span = abs(x2 - x1)
            if x_span < w * 0.30:
                continue
            y_mid = (y1 + y2) / 2.0
            horizontals.append({"line": (x1, y1, x2, y2), "pos": y_mid, "length": length})

        elif angle > 75:                            # near-vertical
            # Require the line to span ≥30 % of image height
            y_span = abs(y2 - y1)
            if y_span < h * 0.30:
                continue
            x_mid = (x1 + x2) / 2.0
            verticals.append({"line": (x1, y1, x2, y2), "pos": x_mid, "length": length})

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    def cluster_lines(items: list[dict], pos_tol: int = 25) -> list[dict]:
        items = sorted(items, key=lambda d: d["pos"])
        clusters: list[list[dict]] = []
        for item in items:
            if not clusters or abs(item["pos"] - clusters[-1][-1]["pos"]) > pos_tol:
                clusters.append([item])
            else:
                clusters[-1].append(item)

        result = []
        for cluster in clusters:
            total_len = sum(x["length"] for x in cluster)
            avg_pos   = sum(x["pos"] * x["length"] for x in cluster) / total_len
            best_line = max(cluster, key=lambda x: x["length"])
            result.append({"pos": avg_pos, "weight": total_len, "line": best_line["line"]})
        return result

    h_clusters = cluster_lines(horizontals, pos_tol=25)
    v_clusters = cluster_lines(verticals,   pos_tol=25)

    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None

    top    = min(h_clusters, key=lambda c: c["pos"])
    bottom = max(h_clusters, key=lambda c: c["pos"])
    left   = min(v_clusters, key=lambda c: c["pos"])
    right  = max(v_clusters, key=lambda c: c["pos"])

    def intersect(l1: tuple, l2: tuple) -> list[float] | None:
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-8:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return [px, py]

    tl = intersect(top["line"],    left["line"])
    tr = intersect(top["line"],    right["line"])
    br = intersect(bottom["line"], right["line"])
    bl = intersect(bottom["line"], left["line"])

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = order_corners(np.array([tl, tr, br, bl], dtype=np.float32))

    if not is_convex_quad(corners):
        return None

    score = score_candidate(corners, image_small.shape)
    if score <= 0:
        return None

    # Small bonus for clusters backed by a lot of evidence
    total_weight = sum(c["weight"] for c in [top, bottom, left, right])
    score += 0.05 * min(1.0, total_weight / 4000.0)

    area = quad_area(corners)
    return CandidateQuad(corners=corners, area=area, score=float(score), method="lines")


# ---------------------------------------------------------------------------
# Detection method 3: GrabCut + RANSAC (new)
# ---------------------------------------------------------------------------

def _ransac_line(points: np.ndarray, n_iter: int = 600, thresh: float = 8.0) -> tuple | None:
    """
    Fit a line  ax + by + c = 0  (unit normal) to *points* via RANSAC.
    Returns (a, b, c) or None if fewer than 2 points are supplied.
    """
    if len(points) < 2:
        return None

    rng  = np.random.default_rng(seed=42)
    best: tuple | None = None
    best_n = 0

    for _ in range(n_iter):
        idx = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        d = p2 - p1
        length = float(np.hypot(d[0], d[1]))
        if length < 1.0:
            continue
        a, b = -d[1] / length, d[0] / length
        c    = -(a * p1[0] + b * p1[1])
        n    = int((np.abs(a * points[:, 0] + b * points[:, 1] + c) < thresh).sum())
        if n > best_n:
            best_n = n
            best   = (float(a), float(b), float(c))

    return best


def _line_intersect(l1: tuple, l2: tuple) -> tuple[float, float] | None:
    """Intersect two lines expressed as (a, b, c) where ax + by + c = 0."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return float(x), float(y)


def detect_grabcut_quad(image_small: np.ndarray, _cfg: ScanConfig) -> CandidateQuad | None:
    """
    Use GrabCut to separate canvas from background, then fit straight lines
    to the four sides of the resulting mask boundary via RANSAC, and intersect
    them to get the four canvas corners.

    Why this works better than pure edge detection for paintings:
    - GrabCut uses colour statistics, not just gradients, so it isn't confused
      by the many interior edges that paint strokes create.
    - RANSAC is robust to the ragged/noisy boundary that GrabCut produces.

    Seeding strategy
    ----------------
    The outermost 3 % of the image is marked as definite background (it almost
    never contains canvas pixels).  The inner 80 % is marked as definite
    foreground to give GrabCut a strong canvas colour model to work from.
    The remaining annular region is left as "probable foreground" for GrabCut
    to classify.
    """
    h, w = image_small.shape[:2]

    # --- 1. Build the GrabCut seed mask ---
    border_frac  = 0.03
    fg_frac      = 0.11
    border_px    = int(min(h, w) * border_frac)
    fg_inset_px  = int(min(h, w) * fg_frac)

    mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
    # Thin outer strip = definite background
    mask[:border_px, :]  = cv2.GC_BGD
    mask[-border_px:, :] = cv2.GC_BGD
    mask[:, :border_px]  = cv2.GC_BGD
    mask[:, -border_px:] = cv2.GC_BGD
    # Inner rectangle = definite foreground
    mask[fg_inset_px:-fg_inset_px, fg_inset_px:-fg_inset_px] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_small, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), np.uint8(255), np.uint8(0)
    )

    # --- 2. Clean up the mask ---
    ksize  = max(15, int(min(h, w) * 0.02))
    ksize += ksize % 2 == 0           # ensure odd isn't needed — MORPH_RECT is fine either way
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)

    # --- 3. Extract boundary pixels ---
    boundary = cv2.Canny(fg_mask, 50, 150)
    yx = np.column_stack(np.where(boundary > 0))   # rows of (y, x)
    if len(yx) < 20:
        return None
    pts_xy = yx[:, [1, 0]].astype(np.float32)     # convert to (x, y)

    # --- 4. Split boundary into four side zones ---
    left_pts   = pts_xy[pts_xy[:, 0] < w * 0.30]
    right_pts  = pts_xy[pts_xy[:, 0] > w * 0.70]
    top_pts    = pts_xy[pts_xy[:, 1] < h * 0.30]
    bottom_pts = pts_xy[pts_xy[:, 1] > h * 0.70]

    min_side_pts = 10
    if any(len(s) < min_side_pts for s in [left_pts, right_pts, top_pts, bottom_pts]):
        return None

    # --- 5. RANSAC line fit for each side ---
    L = _ransac_line(left_pts)
    R = _ransac_line(right_pts)
    T = _ransac_line(top_pts)
    B = _ransac_line(bottom_pts)

    if any(line is None for line in [L, R, T, B]):
        return None

    # --- 6. Intersect side lines to get corners ---
    tl = _line_intersect(T, L)
    tr = _line_intersect(T, R)
    br = _line_intersect(B, R)
    bl = _line_intersect(B, L)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = order_corners(np.array([tl, tr, br, bl], dtype=np.float32))
    score   = score_candidate(corners, image_small.shape)
    if score <= 0:
        return None

    area = quad_area(corners)
    return CandidateQuad(corners=corners, area=area, score=float(score), method="grabcut")


# ---------------------------------------------------------------------------
# Combining all detectors
# ---------------------------------------------------------------------------

def detect_best_quad(image_small: np.ndarray, cfg: ScanConfig) -> CandidateQuad | None:
    """
    Run all three detectors and return the highest-scoring candidate.
    GrabCut is tried first because it is the most expensive; if it already
    produces a high-confidence result the others act as confirmation.
    """
    candidates: list[CandidateQuad] = []

    for detector in [detect_grabcut_quad, detect_contour_quad, detect_line_quad]:
        result = detector(image_small, cfg)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return None

    return max(candidates, key=lambda c: c.score)


# ---------------------------------------------------------------------------
# Warp / output helpers (unchanged)
# ---------------------------------------------------------------------------

def compute_output_size(corners: np.ndarray) -> tuple[int, int]:
    rect = order_corners(corners)
    tl, tr, br, bl = rect

    width_a  = np.linalg.norm(br - bl)
    width_b  = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width  = max(1, int(round(max(width_a,  width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))
    return max_width, max_height


def warp_from_quad(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    rect = order_corners(corners)
    max_width, max_height = compute_output_size(rect)

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def trim_border(image: np.ndarray, trim_px: int) -> np.ndarray:
    if trim_px <= 0:
        return image
    h, w = image.shape[:2]
    t, l = trim_px, trim_px
    b, r = h - trim_px, w - trim_px
    if r <= l or b <= t:
        return image
    return image[t:b, l:r].copy()


def default_inset_quad(image: np.ndarray, inset_ratio: float = 0.10) -> np.ndarray:
    h, w = image.shape[:2]
    dx = int(round(w * inset_ratio))
    dy = int(round(h * inset_ratio))
    return np.array(
        [[dx, dy], [w - dx - 1, dy], [w - dx - 1, h - dy - 1], [dx, h - dy - 1]],
        dtype=np.float32,
    )


def resize_to_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w    = image.shape[:2]
    largest = max(h, w)
    if largest <= 0:
        raise ValueError("Invalid image dimensions")
    if largest == max_dim:
        return image.copy()
    scale  = max_dim / float(largest)
    new_w  = max(1, int(round(w * scale)))
    new_h  = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def build_derived_output_path(src_path: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{src_path.stem}{suffix}.jpg"


def save_master_and_derivatives(
    warped: np.ndarray,
    src_path: Path,
    out_dir: Path,
    jpg_quality: int,
    output_stem: str | None = None,
) -> tuple[Path, Path, Path]:
    """Save master, 600px and 140px derivatives.

    If output_stem is given (e.g. "SA221a"), output files are named
    SA221a_master.jpg etc.  Otherwise src_path.stem is used.
    """
    stem        = output_stem if output_stem is not None else src_path.stem
    master_path = out_dir / f"{stem}_master.jpg"
    path_600    = out_dir / f"{stem}_600.jpg"
    path_140    = out_dir / f"{stem}_140.jpg"

    save_jpg(master_path, warped, quality=jpg_quality)
    save_jpg(path_600,    resize_to_max_dim(warped, 600), quality=jpg_quality)
    save_jpg(path_140,    resize_to_max_dim(warped, 140), quality=jpg_quality)

    return master_path, path_600, path_140


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

# A detection that scores above this threshold is accepted automatically
# without opening the interactive corner editor.
_AUTO_ACCEPT_THRESHOLD = 0.92

# A detection that scores below this threshold is too unreliable to use as
# the interactive editor's starting quad — fall back to the inset rectangle.
_INTERACTIVE_SEED_THRESHOLD = 0.55


def process_image(path: Path, out_dir: Path, cfg: ScanConfig, output_stem: str | None = None) -> ProcessResult:
    try:
        image_full  = load_image(path)
        image_small, scale = resize_for_preview(image_full, cfg.downscale_max_dim)

        candidate = detect_best_quad(image_small, cfg)

        if candidate is not None:
            initial_corners_small = candidate.corners.copy()
            initial_method        = candidate.method
            initial_score         = candidate.score
        else:
            initial_corners_small = default_inset_quad(image_small, inset_ratio=0.10)
            initial_method        = "fallback"
            initial_score         = None

        # If the detection is too uncertain, seed the interactive editor with
        # the simple inset quad — it's always closer to the truth than a bad guess.
        if initial_score is not None and initial_score < _INTERACTIVE_SEED_THRESHOLD:
            initial_corners_small = default_inset_quad(image_small, inset_ratio=0.10)

        used_interactive = False

        if cfg.interactive and (initial_score is None or initial_score < _AUTO_ACCEPT_THRESHOLD):
            used_interactive     = True
            edited_corners_small = edit_quad(image_small, initial_corners_small)
            if edited_corners_small is None:
                return ProcessResult(
                    source=path, output_master=None,
                    confidence=initial_score, method=initial_method,
                    accepted=False, used_interactive=True,
                    message="Cancelled by user",
                )
            final_corners_small = edited_corners_small
        else:
            if candidate is None:
                return ProcessResult(
                    source=path, output_master=None,
                    confidence=None, method=None,
                    accepted=False, used_interactive=False,
                    message="No quadrilateral detected",
                )
            final_corners_small = initial_corners_small

        corners_full = scale_points(final_corners_small, scale)
        warped       = warp_from_quad(image_full, corners_full)
        warped       = trim_border(warped, cfg.trim_px)

        out_path, out_600, out_140 = save_master_and_derivatives(
            warped=warped, src_path=path, out_dir=out_dir,
            jpg_quality=cfg.jpg_quality, output_stem=output_stem,
        )

        # --- Edge-map session (--edgemap) ---
        if cfg.edgemap:
            stem = output_stem if output_stem is not None else path.stem
            edgemap_result = edit_edgemap(
                warped,
                l_lo=cfg.canny_lo,
                l_hi=cfg.canny_hi,
            )
            if edgemap_result is not None:
                edges_full, thresholds = edgemap_result
                edges_path     = out_dir / f"{stem}_edges.jpg"
                edges_600_path = out_dir / f"{stem}_edges_600.jpg"
                save_jpg(edges_path,     edges_full,                        quality=cfg.jpg_quality)
                save_jpg(edges_600_path, resize_to_max_dim(edges_full, 600), quality=cfg.jpg_quality)
                print(
                    f"[INFO] Edge map saved — "
                    f"L:{thresholds[0]}/{thresholds[1]}  "
                    f"a:{thresholds[2]}/{thresholds[3]}  "
                    f"b:{thresholds[4]}/{thresholds[5]}"
                )
            else:
                print("[INFO] Edge map session cancelled — no edge files written.")

        if cfg.debug:
            debug_overlay = draw_quad(image_small, final_corners_small)
            save_jpg(out_dir / f"{path.stem}_debug_overlay.jpg", debug_overlay, quality=cfg.jpg_quality)

            gray      = to_gray(image_small)
            gray_blur = blur_gray(gray, cfg.blur_ksize)
            edges     = edge_map(gray_blur, cfg.canny_lo, cfg.canny_hi)
            save_jpg(
                out_dir / f"{path.stem}_debug_edges.jpg",
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                quality=cfg.jpg_quality,
            )

            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                threshold=100, minLineLength=150, maxLineGap=25,
            )
            line_img = image_small.copy()
            if lines is not None:
                for x1, y1, x2, y2 in lines.reshape(-1, 4):
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            save_jpg(out_dir / f"{path.stem}_debug_lines.jpg", line_img, quality=cfg.jpg_quality)

        return ProcessResult(
            source=path, output_master=out_path,
            confidence=initial_score, method=initial_method,
            accepted=True, used_interactive=used_interactive,
            message="OK",
        )

    except Exception as e:
        return ProcessResult(
            source=path, output_master=None,
            confidence=None, method=None,
            accepted=False, used_interactive=False,
            message=f"ERROR: {e}",
        )