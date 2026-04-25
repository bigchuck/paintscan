from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from utils import compute_lab_edges


# ===========================================================================
# Corner editor  (unchanged)
# ===========================================================================

WINDOW_NAME = "paintscan - adjust corners"


@dataclass
class EditorState:
    image: np.ndarray
    corners: np.ndarray
    original_corners: np.ndarray
    drag_index: Optional[int] = None
    accepted: bool = False
    cancelled: bool = False


def clamp_point(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return x, y


def hit_test_corner(corners: np.ndarray, x: int, y: int, radius: int = 14) -> Optional[int]:
    for i, (cx, cy) in enumerate(corners):
        if (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2:
            return i
    return None


def draw_editor_frame(
    image: np.ndarray,
    corners: np.ndarray,
    message: str = "A=accept  R=reset  Q/Esc=quit",
) -> np.ndarray:
    frame = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    corner_colors = [(0, 0, 255), (0, 128, 255), (255, 0, 255), (255, 255, 0)]
    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(frame, (x, y), 8, corner_colors[i % len(corner_colors)], -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)
        cv2.putText(frame, str(i), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (8, 8), (540, 40), (0, 0, 0), -1)
    cv2.putText(frame, message, (14, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    state: EditorState = param
    h, w = state.image.shape[:2]
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = hit_test_corner(state.corners, x, y, radius=16)
        if idx is not None:
            state.drag_index = idx
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drag_index is not None:
            x, y = clamp_point(x, y, w, h)
            state.corners[state.drag_index] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        if state.drag_index is not None:
            x, y = clamp_point(x, y, w, h)
            state.corners[state.drag_index] = [x, y]
            state.drag_index = None


def edit_quad(image: np.ndarray, initial_corners: np.ndarray) -> np.ndarray | None:
    state = EditorState(
        image=image.copy(),
        corners=initial_corners.astype(np.float32).copy(),
        original_corners=initial_corners.astype(np.float32).copy(),
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback, state)
    try:
        while True:
            frame = draw_editor_frame(state.image, state.corners)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("a"), ord("A")):
                state.accepted = True
                break
            elif key in (ord("r"), ord("R")):
                state.corners = state.original_corners.copy()
                state.drag_index = None
            elif key in (ord("q"), ord("Q"), 27):
                state.cancelled = True
                break
        cv2.destroyWindow(WINDOW_NAME)
    except Exception:
        cv2.destroyWindow(WINDOW_NAME)
        raise
    if state.accepted:
        return state.corners.astype(np.float32)
    return None


# ===========================================================================
# Edge-map interactive editor
# ===========================================================================

_EDGEMAP_WINDOW = "paintscan - edge map  (T=take  R=reset  Esc=done/exit-local)"

# --- Layout constants -------------------------------------------------------
_CTRL_W    = 300
_DISPLAY_H = 800
_BORDER    = 10

_CTRL_TOP  = 38
_BAND_H    = 110
_TRACK_X0  = 28
_TRACK_X1  = 272
_HANDLE_R  = 9

_BTN_W, _BTN_H = 72, 34
_BTN_CY        = _CTRL_TOP + 6 * _BAND_H + 48   # ≈ 746
_BTN_RESET_CX  = 50
_BTN_TAKE_CX   = 150
_BTN_DONE_CX   = 250

_TAKE_COUNT_Y  = _BTN_CY + _BTN_H // 2 + 22     # ≈ 785

_OC_BTN_W   = 116
_OC_BTN_H   = 28
_OC_BTN_CY  = 820
_BTN_OVL_CX = 75
_BTN_CLR_CX = 220

_TOTAL_H = 880   # extended by 30 px to fit the SEAL row

# Amber accent for local mode (BGR)
_AMBER: tuple = (0, 165, 255)

# SEAL row (local mode only) — gap-close control sits below OVL/CLR
_SEAL_ROW_Y   = 856
_SEAL_MINUS_CX = 60
_SEAL_PLUS_CX  = 240
_SEAL_PBTN_W   = 38
_SEAL_PBTN_H   = 24
_SEAL_MIN      = 0
_SEAL_MAX      = 20
_SEAL_DEFAULT  = 4    # bridges gaps up to ~8 px wide; raise for loose edge maps

# --- Edge colour palette ----------------------------------------------------
_EDGE_COLORS: list[tuple[str, tuple, tuple]] = [
    ("BLK",  (  0,   0,   0), (255, 255, 255)),
    ("WHT",  (255, 255, 255), (  0,   0,   0)),
    ("CYAN", (255, 255,   0), ( 48,  48,  48)),
    ("RED",  (  0,   0, 255), ( 48,  48,  48)),
    ("YEL",  (  0, 255, 255), ( 48,  48,  48)),
]

# Per-slider definitions: (label, min, max, default, BGR colour)
_SLIDER_DEFS: list[tuple] = [
    ("L* lo",  0, 255,  50, (180, 180, 180)),
    ("L* hi",  0, 255, 150, (220, 220, 220)),
    ("a* lo",  0, 255,  30, ( 80, 200,  80)),
    ("a* hi",  0, 255,  90, (100, 230, 100)),
    ("b* lo",  0, 255,  30, ( 30, 200, 220)),
    ("b* hi",  0, 255,  90, ( 50, 230, 240)),
]

# Local-mode slider label / handle colour variants (amber family)
_LOCAL_SLIDER_META: list[tuple] = [
    ("L* lo \u24db", ( 80, 160, 255)),
    ("L* hi \u24db", (100, 180, 255)),
    ("a* lo \u24db", ( 60, 200, 220)),
    ("a* hi \u24db", ( 80, 220, 240)),
    ("b* lo \u24db", ( 40, 180, 255)),
    ("b* hi \u24db", ( 60, 200, 255)),
]


# ---------------------------------------------------------------------------
# Slider
# ---------------------------------------------------------------------------

@dataclass
class _Slider:
    label:   str
    min_val: int
    max_val: int
    value:   int
    color:   tuple

    def handle_x(self) -> int:
        span  = max(1, self.max_val - self.min_val)
        ratio = (self.value - self.min_val) / span
        return int(_TRACK_X0 + ratio * (_TRACK_X1 - _TRACK_X0))

    def value_from_x(self, px: int) -> int:
        span  = _TRACK_X1 - _TRACK_X0
        ratio = (px - _TRACK_X0) / max(1, span)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self.min_val + ratio * (self.max_val - self.min_val)))


def _track_y(idx: int) -> int:
    return _CTRL_TOP + idx * _BAND_H + 58


def _make_local_sliders(global_sliders: list) -> list:
    """Create local sliders seeded from the current global values."""
    return [
        _Slider(
            label   = _LOCAL_SLIDER_META[i][0],
            min_val = defn[1],
            max_val = defn[2],
            value   = gl.value,
            color   = _LOCAL_SLIDER_META[i][1],
        )
        for i, (defn, gl) in enumerate(zip(_SLIDER_DEFS, global_sliders))
    ]


# ---------------------------------------------------------------------------
# EdgemapState
# ---------------------------------------------------------------------------

@dataclass
class _EdgemapState:
    warped_full:    np.ndarray
    warped_display: np.ndarray
    master_panel:   np.ndarray
    sliders:        list
    initial_values: list
    drag_idx:       Optional[int] = None
    done:           bool = False
    takes:          list = field(default_factory=list)
    color_idx:      int  = 0
    overlay_on:     bool = False
    edges_dirty:    bool = True
    edge_panel:     Optional[np.ndarray] = None
    overlay_panel:  Optional[np.ndarray] = None
    # Cached raw inv_gray at display resolution — fed to the flood filler
    inv_gray_cache: Optional[np.ndarray] = None

    # Local region
    local_mode:       bool = False
    local_mask:       Optional[np.ndarray] = None   # uint8, display-res; 255=selected
    local_bbox:       Optional[tuple] = None         # (x0,y0,x1,y1) display coords
    local_seed_disp:  Optional[tuple] = None         # (x,y) display coords
    local_seal:       int  = _SEAL_DEFAULT           # gap-close radius before flood fill
    local_sliders:    list = field(default_factory=list)
    local_init_vals:  list = field(default_factory=list)
    local_drag_idx:   Optional[int] = None
    local_dirty:      bool = False
    local_edge_panel: Optional[np.ndarray] = None
    local_zoom_panel: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Control panel drawing
# ---------------------------------------------------------------------------

def _draw_ctrl_panel(
    sliders: list,
    take_count: int = 0,
    color_idx: int = 0,
    overlay_on: bool = False,
    local_mode: bool = False,
    local_sliders: list | None = None,
    local_seal: int = _SEAL_DEFAULT,
) -> np.ndarray:
    panel = np.full((_TOTAL_H, _CTRL_W, 3), 28, dtype=np.uint8)

    # Title bar
    if local_mode:
        title_bg   = (0, 100, 180)
        title_text = "LOCAL MODE"
    else:
        title_bg   = (45, 45, 45)
        title_text = "Lab Edge Thresholds"
    cv2.rectangle(panel, (0, 0), (_CTRL_W, _CTRL_TOP - 4), title_bg, -1)
    cv2.putText(panel, title_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    active = local_sliders if (local_mode and local_sliders) else sliders

    for i, sl in enumerate(active):
        ty      = _track_y(i)
        label_y = ty - 26
        value_y = ty + 24
        if i % 2 == 0 and i > 0:
            sep_y = _CTRL_TOP + i * _BAND_H - 6
            cv2.line(panel, (10, sep_y), (_CTRL_W - 10, sep_y), (60, 60, 60), 1)
        cv2.putText(panel, sl.label, (_TRACK_X0, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, sl.color, 1, cv2.LINE_AA)
        cv2.line(panel, (_TRACK_X0, ty), (_TRACK_X1, ty), (90, 90, 90), 2)
        for tx in (_TRACK_X0, _TRACK_X1):
            cv2.line(panel, (tx, ty - 5), (tx, ty + 5), (70, 70, 70), 1)
        hx = sl.handle_x()
        cv2.circle(panel, (hx, ty), _HANDLE_R, sl.color, -1)
        cv2.circle(panel, (hx, ty), _HANDLE_R + 1, (240, 240, 240), 1)
        val_str    = str(sl.value)
        (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(panel, val_str, (_TRACK_X1 - tw, value_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    # Main buttons
    rst_col = (0, 100, 180) if local_mode else (90, 90, 30)
    _draw_button(panel, "RESET", _BTN_RESET_CX, _BTN_CY, rst_col, _BTN_W, _BTN_H)
    _draw_button(panel, "TAKE",  _BTN_TAKE_CX,  _BTN_CY, (35, 130, 35), _BTN_W, _BTN_H)
    _draw_button(panel, "DONE",  _BTN_DONE_CX,  _BTN_CY, (70,  70, 70), _BTN_W, _BTN_H)

    if take_count == 0:
        cnt_text, cnt_col = "taken: 0", (90, 90, 90)
    else:
        cnt_text, cnt_col = f"taken: {take_count}", (60, 200, 60)
    (tw, _), _ = cv2.getTextSize(cnt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(panel, cnt_text, (_BTN_TAKE_CX - tw // 2, _TAKE_COUNT_Y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, cnt_col, 1, cv2.LINE_AA)

    # Second row: OVL / EXIT LOCAL  +  CLR
    if local_mode:
        _draw_button(panel, "EXIT LOCAL", _BTN_OVL_CX, _OC_BTN_CY,
                     _AMBER, _OC_BTN_W, _OC_BTN_H)
    else:
        ovl_label = "OVL: ON " if overlay_on else "OVL: OFF"
        ovl_color = (35, 130, 35) if overlay_on else (60, 60, 60)
        _draw_button(panel, ovl_label, _BTN_OVL_CX, _OC_BTN_CY,
                     ovl_color, _OC_BTN_W, _OC_BTN_H)

    clr_label = f"CLR: {_EDGE_COLORS[color_idx][0]}"
    _draw_button(panel, clr_label, _BTN_CLR_CX, _OC_BTN_CY,
                 (55, 55, 90), _OC_BTN_W, _OC_BTN_H)

    # Hint (global) / SEAL control (local)
    if local_mode:
        # − button
        _draw_button(panel, "-", _SEAL_MINUS_CX, _SEAL_ROW_Y,
                     (60, 60, 80), _SEAL_PBTN_W, _SEAL_PBTN_H)
        # + button
        _draw_button(panel, "+", _SEAL_PLUS_CX, _SEAL_ROW_Y,
                     (60, 60, 80), _SEAL_PBTN_W, _SEAL_PBTN_H)
        # Label + value
        seal_text = f"SEAL: {local_seal}"
        (tw, _), _ = cv2.getTextSize(seal_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(panel, seal_text,
                    (_CTRL_W // 2 - tw // 2, _SEAL_ROW_Y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, _AMBER, 1, cv2.LINE_AA)
    else:
        hint = "Click edge panel to select area"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(panel, hint, (_CTRL_W // 2 - tw // 2, _TOTAL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    return panel


def _draw_button(panel, label, cx, cy, color, btn_w=_BTN_W, btn_h=_BTN_H):
    x0, y0 = cx - btn_w // 2, cy - btn_h // 2
    x1, y1 = cx + btn_w // 2, cy + btn_h // 2
    cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(panel, label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _hit_button(x, y, cx, cy, btn_w=_BTN_W, btn_h=_BTN_H):
    return abs(x - cx) <= btn_w // 2 and abs(y - cy) <= btn_h // 2


# ---------------------------------------------------------------------------
# Image panel helpers
# ---------------------------------------------------------------------------

def _add_border(image: np.ndarray, px: int, color=(255, 255, 255)) -> np.ndarray:
    return cv2.copyMakeBorder(image, px, px, px, px,
                              cv2.BORDER_CONSTANT, value=color)


def _make_edge_panel(
    warped_display: np.ndarray,
    sliders: list,
    color_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bordered_BGR_panel, inv_gray_cache)."""
    vals     = [sl.value for sl in sliders]
    inv_gray = compute_lab_edges(warped_display, *vals)
    _name, edge_bgr, bg_bgr = _EDGE_COLORS[color_idx]
    out = np.full((*inv_gray.shape, 3), bg_bgr, dtype=np.uint8)
    out[inv_gray == 0] = edge_bgr
    return _add_border(out, _BORDER), inv_gray


def _make_overlay_panel(warped_display, sliders, color_idx):
    vals     = [sl.value for sl in sliders]
    inv_gray = compute_lab_edges(warped_display, *vals)
    _name, edge_bgr, _bg = _EDGE_COLORS[color_idx]
    out = warped_display.copy()
    out[inv_gray == 0] = edge_bgr
    return _add_border(out, _BORDER)


def _scale_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image.copy()
    scale  = target_h / h
    new_w  = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, target_h), interpolation=interp)


def _maximize_window(title: str) -> None:
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 3)
            return
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        cv2.resizeWindow(title, sw, sh - 60)
    except Exception:
        pass


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    dh = h - img.shape[0]
    if dh <= 0:
        return img
    return np.vstack([img, np.full((dh, img.shape[1], 3), 28, dtype=np.uint8)])


# ---------------------------------------------------------------------------
# Local region helpers
# ---------------------------------------------------------------------------

def _flood_fill_region(
    inv_gray: np.ndarray,
    seed_x: int,
    seed_y: int,
    seal_px: int = 0,
) -> np.ndarray:
    """
    Flood-fill from (seed_x, seed_y) through connected white (non-edge)
    pixels in *inv_gray* (255=background, 0=edge).

    *seal_px* controls a morphological closing pass applied to *inv_gray*
    before the fill.  A kernel of (2*seal_px+1) × (2*seal_px+1) bridges
    gaps up to roughly 2*seal_px pixels wide, preventing the fill from
    leaking through small openings in the edge map.  seal_px=0 = no closing.

    If the seed lands on an edge pixel the function snaps to the nearest
    non-edge pixel within a small radius before filling.

    Returns a uint8 mask (same shape as inv_gray): 255 = selected, 0 = not.
    """
    h, w = inv_gray.shape

    # Optional edge dilation to bridge gaps in the boundary.
    # Pure dilation (no erosion pass) thickens existing edges so gaps up to
    # 2*seal_px pixels wide are sealed without the bridge being erased by a
    # subsequent erosion (which MORPH_CLOSE would do, destroying thin bridges).
    if seal_px > 0:
        k    = 2 * seal_px + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # Work on edges-as-foreground (255), dilate, convert back
        edges_fg  = cv2.bitwise_not(inv_gray)
        edges_fat = cv2.dilate(edges_fg, kern)
        work_img  = cv2.bitwise_not(edges_fat)
    else:
        work_img = inv_gray

    sx, sy = seed_x, seed_y

    # Snap off edge pixels (check work_img after closing)
    if work_img[sy, sx] == 0:
        found = False
        for r in range(1, 14):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < w and 0 <= ny < h and work_img[ny, nx] > 0:
                        sx, sy = nx, ny
                        found  = True
                        break
                if found:
                    break
            if found:
                break

    ff_img  = work_img.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff_img, ff_mask, (sx, sy), 128,
                  loDiff=10, upDiff=10,
                  flags=4 | cv2.FLOODFILL_MASK_ONLY)
    return (ff_mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _make_local_edge_panel(
    warped_display: np.ndarray,
    global_sliders: list,
    local_sliders: list,
    local_mask: np.ndarray,
    color_idx: int,
) -> np.ndarray:
    """
    Composite edge panel for local mode:
      • Inside selection  → local Lab thresholds, full brightness
      • Outside selection → global Lab thresholds, dimmed to 35 %
      • Selection boundary → 2-px amber ring
      • Top banner → amber bar: "LOCAL MODE — Esc or click outside to exit"
    """
    g_vals = [sl.value for sl in global_sliders]
    l_vals = [sl.value for sl in local_sliders]
    global_inv = compute_lab_edges(warped_display, *g_vals)
    local_inv  = compute_lab_edges(warped_display, *l_vals)

    merged = global_inv.copy()
    merged[local_mask > 0] = local_inv[local_mask > 0]

    _name, edge_bgr, bg_bgr = _EDGE_COLORS[color_idx]
    out = np.full((*merged.shape, 3), bg_bgr, dtype=np.uint8)
    out[merged == 0] = edge_bgr

    # Dim outside-mask area
    outside = local_mask == 0
    out[outside] = (out[outside].astype(np.float32) * 0.35).astype(np.uint8)

    # Amber boundary ring
    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(local_mask, kernel, iterations=1)
    ring    = (dilated > 0) & (local_mask == 0)
    out[ring] = _AMBER

    # LOCAL MODE banner
    banner_h = 28
    cv2.rectangle(out, (0, 0), (out.shape[1], banner_h), _AMBER, -1)
    cv2.putText(out,
                "LOCAL MODE  \u2014  Esc or click outside selection to exit",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (255, 255, 255), 1, cv2.LINE_AA)

    return _add_border(out, _BORDER, color=_AMBER)


def _make_zoom_panel(
    warped_display: np.ndarray,
    local_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop *warped_display* to *local_bbox* (+ 15 % padding) and scale to
    _DISPLAY_H.  Bordered in amber to signal zoomed / local context.
    """
    x0, y0, x1, y1 = local_bbox
    h, w = warped_display.shape[:2]
    pad_x = max(10, int((x1 - x0) * 0.15))
    pad_y = max(10, int((y1 - y0) * 0.15))
    cx0 = max(0, x0 - pad_x);  cy0 = max(0, y0 - pad_y)
    cx1 = min(w, x1 + pad_x);  cy1 = min(h, y1 + pad_y)
    crop   = warped_display[cy0:cy1, cx0:cx1].copy()
    zoomed = _scale_to_height(crop, _DISPLAY_H)
    cv2.rectangle(zoomed, (0, 0), (64, 22), _AMBER, -1)
    cv2.putText(zoomed, "ZOOM", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return _add_border(zoomed, _BORDER, color=_AMBER)


# ---------------------------------------------------------------------------
# Local mode transitions
# ---------------------------------------------------------------------------

def _enter_local_mode(state: _EdgemapState, disp_x: int, disp_y: int) -> None:
    if state.inv_gray_cache is None:
        return
    mask = _flood_fill_region(state.inv_gray_cache, disp_x, disp_y,
                               seal_px=state.local_seal)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return
    state.local_mask      = mask
    state.local_bbox      = bbox
    state.local_seed_disp = (disp_x, disp_y)
    state.local_sliders   = _make_local_sliders(state.sliders)
    state.local_init_vals = [sl.value for sl in state.local_sliders]
    state.local_mode      = True
    state.local_dirty     = True


def _rerun_flood_fill(state: _EdgemapState) -> None:
    """Re-run the flood fill from the same seed with the current seal value.

    Called when SEAL changes while already in local mode.  Unlike
    _enter_local_mode, this preserves the local slider values the user has
    already adjusted — only the mask and bbox are recomputed.
    """
    if state.inv_gray_cache is None or state.local_seed_disp is None:
        return
    sx, sy = state.local_seed_disp
    mask   = _flood_fill_region(state.inv_gray_cache, sx, sy,
                                 seal_px=state.local_seal)
    bbox   = _bbox_from_mask(mask)
    if bbox is None:
        return
    state.local_mask  = mask
    state.local_bbox  = bbox
    state.local_dirty = True


def _exit_local_mode(state: _EdgemapState) -> None:
    state.local_mode      = False
    state.local_mask      = None
    state.local_bbox      = None
    state.local_seed_disp = None
    state.local_sliders   = []
    state.local_init_vals = []
    state.local_drag_idx  = None
    state.local_edge_panel = None
    state.local_zoom_panel = None


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def _edgemap_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    state: _EdgemapState = param
    ep_w = state.warped_display.shape[1] + 2 * _BORDER
    on_edge_panel = (_CTRL_W <= x < _CTRL_W + ep_w)

    if event == cv2.EVENT_LBUTTONDOWN:

        if x < _CTRL_W:
            # ---- Control panel ------------------------------------------ #
            if state.local_mode:
                for i, sl in enumerate(state.local_sliders):
                    ty = _track_y(i)
                    if (x - sl.handle_x()) ** 2 + (y - ty) ** 2 <= (_HANDLE_R + 5) ** 2:
                        state.local_drag_idx = i
                        return
                if _hit_button(x, y, _BTN_RESET_CX, _BTN_CY):
                    for sl, iv in zip(state.local_sliders, state.local_init_vals):
                        sl.value = iv
                    state.local_dirty = True
                elif _hit_button(x, y, _BTN_TAKE_CX, _BTN_CY):
                    _do_take(state)
                elif _hit_button(x, y, _BTN_DONE_CX, _BTN_CY):
                    state.done = True
                elif _hit_button(x, y, _BTN_OVL_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    _exit_local_mode(state)
                    state.edges_dirty = True
                elif _hit_button(x, y, _BTN_CLR_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    state.color_idx   = (state.color_idx + 1) % len(_EDGE_COLORS)
                    state.local_dirty = True
                elif _hit_button(x, y, _SEAL_MINUS_CX, _SEAL_ROW_Y,
                                 _SEAL_PBTN_W, _SEAL_PBTN_H):
                    state.local_seal = max(_SEAL_MIN, state.local_seal - 1)
                    _rerun_flood_fill(state)
                elif _hit_button(x, y, _SEAL_PLUS_CX, _SEAL_ROW_Y,
                                 _SEAL_PBTN_W, _SEAL_PBTN_H):
                    state.local_seal = min(_SEAL_MAX, state.local_seal + 1)
                    _rerun_flood_fill(state)
            else:
                for i, sl in enumerate(state.sliders):
                    ty = _track_y(i)
                    if (x - sl.handle_x()) ** 2 + (y - ty) ** 2 <= (_HANDLE_R + 5) ** 2:
                        state.drag_idx = i
                        return
                if _hit_button(x, y, _BTN_RESET_CX, _BTN_CY):
                    for sl, iv in zip(state.sliders, state.initial_values):
                        sl.value = iv
                    state.edges_dirty = True
                elif _hit_button(x, y, _BTN_TAKE_CX, _BTN_CY):
                    _do_take(state)
                elif _hit_button(x, y, _BTN_DONE_CX, _BTN_CY):
                    state.done = True
                elif _hit_button(x, y, _BTN_OVL_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    state.overlay_on  = not state.overlay_on
                    state.edges_dirty = True
                elif _hit_button(x, y, _BTN_CLR_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    state.color_idx   = (state.color_idx + 1) % len(_EDGE_COLORS)
                    state.edges_dirty = True

        elif on_edge_panel:
            # ---- Edge panel click ---------------------------------------- #
            dh, dw = state.warped_display.shape[:2]
            disp_x = max(0, min(x - _CTRL_W - _BORDER, dw - 1))
            disp_y = max(0, min(y - _BORDER, dh - 1))

            if state.local_mode:
                # Click outside selection → exit local mode
                if state.local_mask is not None and state.local_mask[disp_y, disp_x] == 0:
                    _exit_local_mode(state)
                    state.edges_dirty = True
            else:
                # Enter local mode via flood fill
                _enter_local_mode(state, disp_x, disp_y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.local_mode and state.local_drag_idx is not None:
            sl = state.local_sliders[state.local_drag_idx]
            nv = sl.value_from_x(x)
            if nv != sl.value:
                sl.value          = nv
                state.local_dirty = True
        elif not state.local_mode and state.drag_idx is not None:
            sl = state.sliders[state.drag_idx]
            nv = sl.value_from_x(x)
            if nv != sl.value:
                sl.value          = nv
                state.edges_dirty = True

    elif event == cv2.EVENT_LBUTTONUP:
        state.drag_idx       = None
        state.local_drag_idx = None


# ---------------------------------------------------------------------------
# Take helper
# ---------------------------------------------------------------------------

def _do_take(state: _EdgemapState) -> None:
    """
    Compute a full-resolution edge map and append a 3-tuple to state.takes:

        (edges_full, global_thresholds, local_info)

    *global_thresholds* : (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi)
    *local_info*        : None for global-only takes, or::

        {
            "seed":       (x, y),                        # full-res pixels
            "bbox":       (x0, y0, x1, y1),              # full-res pixels
            "thresholds": (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi),
        }

    The saved edge map is always raw grayscale regardless of display colour.
    """
    global_vals    = tuple(sl.value for sl in state.sliders)
    h_full, w_full = state.warped_full.shape[:2]
    h_disp, w_disp = state.warped_display.shape[:2]

    if state.local_mode and state.local_mask is not None:
        local_vals  = tuple(sl.value for sl in state.local_sliders)
        global_disp = compute_lab_edges(state.warped_display, *global_vals)
        local_disp  = compute_lab_edges(state.warped_display, *local_vals)
        merged      = global_disp.copy()
        merged[state.local_mask > 0] = local_disp[state.local_mask > 0]
        full_edges = cv2.resize(merged, (w_full, h_full),
                                interpolation=cv2.INTER_NEAREST)

        rx = w_full / max(1, w_disp)
        ry = h_full / max(1, h_disp)
        sdx, sdy   = state.local_seed_disp
        x0, y0, x1, y1 = state.local_bbox
        local_info: dict | None = {
            "seed":       (int(sdx * rx), int(sdy * ry)),
            "bbox":       (int(x0 * rx), int(y0 * ry),
                           int(x1 * rx), int(y1 * ry)),
            "thresholds": local_vals,
        }
    else:
        disp_edges = compute_lab_edges(state.warped_display, *global_vals)
        full_edges = cv2.resize(disp_edges, (w_full, h_full),
                                interpolation=cv2.INTER_NEAREST)
        local_info = None

    state.takes.append((full_edges, global_vals, local_info))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def edit_edgemap(
    warped: np.ndarray,
    l_lo: int = 50,
    l_hi: int = 150,
    a_lo: int = 30,
    a_hi: int = 90,
    b_lo: int = 30,
    b_hi: int = 90,
) -> list[tuple[np.ndarray, tuple, dict | None]]:
    """
    Open the three-panel Lab edge-map editor.

    Global mode
    -----------
    T / TAKE  = commit current thresholds as a take
    R / RESET = restore sliders to session-open values
    O         = overlay toggle (edges composited onto painting)
    C         = cycle edge colour palette
    Esc/D/Q   = close session and return takes

    Local mode  (entered by clicking anywhere on the edge panel)
    ----------
    Sliders control the *local* Lab thresholds for the flood-filled region.
    T / TAKE      = take with global-outside + local-inside merge
    R / RESET     = reset local sliders to their entry values
    Esc / L       = exit local mode → return to global mode
    EXIT LOCAL btn or click-outside-selection = same as Esc
    D / Q         = close session (exit local mode AND session)

    Returns
    -------
    list of (edges_full_res, global_thresholds, local_info)
        One entry per TAKE.  Empty list if closed without taking.
    """
    initial_vals = [l_lo, l_hi, a_lo, a_hi, b_lo, b_hi]
    sliders = [
        _Slider(
            label   = defn[0],
            min_val = defn[1],
            max_val = defn[2],
            value   = max(defn[1], min(defn[2], iv)),
            color   = defn[4],
        )
        for defn, iv in zip(_SLIDER_DEFS, initial_vals)
    ]

    master_scaled = _scale_to_height(warped, _DISPLAY_H)
    master_panel  = _add_border(master_scaled, _BORDER)

    state = _EdgemapState(
        warped_full    = warped,
        warped_display = master_scaled,
        master_panel   = master_panel,
        sliders        = sliders,
        initial_values = list(initial_vals),
        edges_dirty    = True,
    )

    cv2.namedWindow(_EDGEMAP_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_EDGEMAP_WINDOW, _edgemap_mouse, state)

    # Bootstrap — render once so HWND exists before maximising
    _boot_ctrl = _draw_ctrl_panel(state.sliders, 0, state.color_idx, state.overlay_on)
    state.edge_panel, state.inv_gray_cache = _make_edge_panel(
        state.warped_display, state.sliders, state.color_idx)
    state.edges_dirty = False
    _bt_h = max(_boot_ctrl.shape[0], state.edge_panel.shape[0], master_panel.shape[0])
    cv2.imshow(_EDGEMAP_WINDOW, np.hstack([
        _pad_to_height(_boot_ctrl,       _bt_h),
        _pad_to_height(state.edge_panel, _bt_h),
        _pad_to_height(master_panel,     _bt_h),
    ]))
    cv2.waitKey(1)
    _maximize_window(_EDGEMAP_WINDOW)

    try:
        while not state.done:

            # Rebuild panels when dirty
            if state.edges_dirty and not state.local_mode:
                state.edge_panel, state.inv_gray_cache = _make_edge_panel(
                    state.warped_display, state.sliders, state.color_idx)
                if state.overlay_on:
                    state.overlay_panel = _make_overlay_panel(
                        state.warped_display, state.sliders, state.color_idx)
                state.edges_dirty = False

            if state.local_dirty and state.local_mode:
                state.local_edge_panel = _make_local_edge_panel(
                    state.warped_display, state.sliders, state.local_sliders,
                    state.local_mask, state.color_idx)
                state.local_zoom_panel = _make_zoom_panel(
                    state.warped_display, state.local_bbox)
                state.local_dirty = False

            # Assemble frame
            ctrl_panel = _draw_ctrl_panel(
                state.sliders, len(state.takes), state.color_idx, state.overlay_on,
                local_mode=state.local_mode,
                local_sliders=state.local_sliders if state.local_mode else None,
                local_seal=state.local_seal,
            )
            if state.local_mode:
                mid_panel   = state.local_edge_panel if state.local_edge_panel is not None else state.edge_panel
                right_panel = state.local_zoom_panel if state.local_zoom_panel is not None else state.master_panel
            else:
                mid_panel   = state.edge_panel
                right_panel = state.overlay_panel if state.overlay_on else state.master_panel

            target_h = max(ctrl_panel.shape[0], mid_panel.shape[0], right_panel.shape[0])
            cv2.imshow(_EDGEMAP_WINDOW, np.hstack([
                _pad_to_height(ctrl_panel,  target_h),
                _pad_to_height(mid_panel,   target_h),
                _pad_to_height(right_panel, target_h),
            ]))

            # Key handling
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("t"), ord("T")):
                _do_take(state)

            elif key in (ord("r"), ord("R")):
                if state.local_mode:
                    for sl, iv in zip(state.local_sliders, state.local_init_vals):
                        sl.value = iv
                    state.local_dirty = True
                else:
                    for sl, iv in zip(state.sliders, state.initial_values):
                        sl.value = iv
                    state.edges_dirty = True

            elif key == 27:           # Esc
                if state.local_mode:
                    _exit_local_mode(state)
                    state.edges_dirty = True
                else:
                    state.done = True

            elif key in (ord("l"), ord("L")):
                if state.local_mode:
                    _exit_local_mode(state)
                    state.edges_dirty = True

            elif key in (ord("d"), ord("D"), ord("q"), ord("Q")):
                state.done = True

            elif key in (ord("o"), ord("O")):
                if not state.local_mode:
                    state.overlay_on  = not state.overlay_on
                    state.edges_dirty = True

            elif key in (ord("c"), ord("C")):
                state.color_idx = (state.color_idx + 1) % len(_EDGE_COLORS)
                if state.local_mode:
                    state.local_dirty = True
                else:
                    state.edges_dirty = True

    finally:
        cv2.destroyWindow(_EDGEMAP_WINDOW)

    return state.takes