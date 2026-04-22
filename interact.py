from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from utils import compute_lab_edges


# ===========================================================================
# Corner editor  (unchanged from original)
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

    corner_colors = [
        (0, 0, 255),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(frame, (x, y), 8, corner_colors[i % len(corner_colors)], -1)
        cv2.circle(frame, (x, y), 12, (255, 255, 255), 1)
        cv2.putText(
            frame,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (8, 8), (540, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        message,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

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

_EDGEMAP_WINDOW = "paintscan - edge map  (T=take  Q/Esc=cancel)"

# --- Layout constants -------------------------------------------------------
_CTRL_W    = 300          # width of the custom slider control panel
_DISPLAY_H = 800          # target height for both image panels (before border)
_BORDER    = 10           # white border applied to each image panel (pixels)
_TOTAL_H   = _DISPLAY_H + 2 * _BORDER   # control panel height to match bordered panels

_CTRL_TOP  = 38           # y where the first slider band begins
_BAND_H    = 110          # vertical pixels allocated per slider
_TRACK_X0  = 28           # x of left end of slider track
_TRACK_X1  = 272          # x of right end of slider track
_HANDLE_R  = 9            # radius of the draggable handle circle

# Three buttons spaced evenly across the 300px control panel
_BTN_W, _BTN_H = 72, 34
_BTN_CY        = _CTRL_TOP + 6 * _BAND_H + 48   # 38 + 660 + 48 = 746
_BTN_RESET_CX  = 50
_BTN_TAKE_CX   = 150
_BTN_CANCEL_CX = 250

# Per-slider definitions: (label, min, max, default, BGR color)
_SLIDER_DEFS: list[tuple[str, int, int, int, tuple[int, int, int]]] = [
    ("L* lo",  0, 255,  50, (180, 180, 180)),
    ("L* hi",  0, 255, 150, (220, 220, 220)),
    ("a* lo",  0, 255,  30, ( 80, 200,  80)),
    ("a* hi",  0, 255,  90, (100, 230, 100)),
    ("b* lo",  0, 255,  30, ( 30, 200, 220)),
    ("b* hi",  0, 255,  90, ( 50, 230, 240)),
]


# ---------------------------------------------------------------------------
# Slider data class
# ---------------------------------------------------------------------------

@dataclass
class _Slider:
    label:   str
    min_val: int
    max_val: int
    value:   int
    color:   tuple          # BGR for label / handle

    def handle_x(self) -> int:
        """Pixel x of the handle within the control panel."""
        span  = max(1, self.max_val - self.min_val)
        ratio = (self.value - self.min_val) / span
        return int(_TRACK_X0 + ratio * (_TRACK_X1 - _TRACK_X0))

    def value_from_x(self, px: int) -> int:
        """Map a pixel x position back to a clamped integer value."""
        span  = _TRACK_X1 - _TRACK_X0
        ratio = (px - _TRACK_X0) / max(1, span)
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self.min_val + ratio * (self.max_val - self.min_val)))


def _track_y(idx: int) -> int:
    """Y coordinate of the slider track centre for slider *idx*."""
    return _CTRL_TOP + idx * _BAND_H + 58


# ---------------------------------------------------------------------------
# EdgemapState
# ---------------------------------------------------------------------------

@dataclass
class _EdgemapState:
    warped_full:    np.ndarray          # full-resolution master (for final save)
    warped_display: np.ndarray          # display-scaled BGR for fast edge computation
    master_panel:   np.ndarray          # pre-built right panel (BGR, bordered)
    sliders:        list
    initial_values: list                # slider values at session open — for Reset
    drag_idx:       Optional[int] = None
    accepted:       bool = False
    cancelled:      bool = False
    edges_dirty:    bool = True
    edge_panel:     Optional[np.ndarray] = None   # current bordered BGR edge panel


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_ctrl_panel(sliders: list) -> np.ndarray:
    """Render the left-side control panel at _CTRL_W × _TOTAL_H."""
    panel = np.full((_TOTAL_H, _CTRL_W, 3), 28, dtype=np.uint8)

    # Title bar
    cv2.rectangle(panel, (0, 0), (_CTRL_W, _CTRL_TOP - 4), (45, 45, 45), -1)
    cv2.putText(panel, "Lab Edge Thresholds", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    for i, sl in enumerate(sliders):
        ty      = _track_y(i)
        label_y = ty - 26
        value_y = ty + 24

        # Separator between channel groups (L / a / b)
        if i % 2 == 0 and i > 0:
            sep_y = _CTRL_TOP + i * _BAND_H - 6
            cv2.line(panel, (10, sep_y), (_CTRL_W - 10, sep_y), (60, 60, 60), 1)

        # Label
        cv2.putText(panel, sl.label, (_TRACK_X0, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, sl.color, 1, cv2.LINE_AA)

        # Track
        cv2.line(panel, (_TRACK_X0, ty), (_TRACK_X1, ty), (90, 90, 90), 2)
        for tx in (_TRACK_X0, _TRACK_X1):
            cv2.line(panel, (tx, ty - 5), (tx, ty + 5), (70, 70, 70), 1)

        # Handle
        hx = sl.handle_x()
        cv2.circle(panel, (hx, ty), _HANDLE_R, sl.color, -1)
        cv2.circle(panel, (hx, ty), _HANDLE_R + 1, (240, 240, 240), 1)

        # Numeric value right-aligned to track end
        val_str    = str(sl.value)
        (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(panel, val_str, (_TRACK_X1 - tw, value_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    # Three buttons
    _draw_button(panel, "RESET",  _BTN_RESET_CX,  _BTN_CY, ( 90,  90,  30))
    _draw_button(panel, "TAKE",   _BTN_TAKE_CX,   _BTN_CY, ( 35, 130,  35))
    _draw_button(panel, "CANCEL", _BTN_CANCEL_CX, _BTN_CY, ( 35,  35, 130))

    # Keyboard hint at bottom
    cv2.putText(panel, "T=take  R=reset  Q/Esc=cancel",
                (10, _TOTAL_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)

    return panel


def _draw_button(panel: np.ndarray, label: str, cx: int, cy: int,
                 color: tuple) -> None:
    x0, y0 = cx - _BTN_W // 2, cy - _BTN_H // 2
    x1, y1 = cx + _BTN_W // 2, cy + _BTN_H // 2
    cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(panel, label, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _hit_button(x: int, y: int, cx: int, cy: int) -> bool:
    return abs(x - cx) <= _BTN_W // 2 and abs(y - cy) <= _BTN_H // 2


# ---------------------------------------------------------------------------
# Image panel helpers
# ---------------------------------------------------------------------------

def _add_border(image: np.ndarray, px: int) -> np.ndarray:
    """Add a white border of *px* pixels on all four sides."""
    return cv2.copyMakeBorder(
        image, px, px, px, px,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def _make_edge_panel(warped_display: np.ndarray, sliders: list) -> np.ndarray:
    """Compute the inverted Lab edge map and return it as a bordered BGR panel."""
    vals     = [sl.value for sl in sliders]
    inv_gray = compute_lab_edges(warped_display, *vals)
    bgr      = cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR)
    return _add_border(bgr, _BORDER)


def _scale_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image.copy()
    scale  = target_h / h
    new_w  = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, target_h), interpolation=interp)


def _maximize_window(title: str) -> None:
    """
    Ask the OS to maximize *title* into the work area (screen minus taskbar).
    """
    try:
        import ctypes
        SW_MAXIMIZE = 3
        hwnd = ctypes.windll.user32.FindWindowW(None, title)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)
            return
    except Exception:
        pass

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        cv2.resizeWindow(title, sw, sh - 60)
    except Exception:
        pass


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    """Pad bottom of *img* with dark fill to reach height *h* (no-op if already tall enough)."""
    dh = h - img.shape[0]
    if dh <= 0:
        return img
    pad = np.full((dh, img.shape[1], 3), 28, dtype=np.uint8)
    return np.vstack([img, pad])


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def _edgemap_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    state: _EdgemapState = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if x < _CTRL_W:
            # Hit-test slider handles first
            for i, sl in enumerate(state.sliders):
                ty = _track_y(i)
                hx = sl.handle_x()
                if (x - hx) ** 2 + (y - ty) ** 2 <= (_HANDLE_R + 5) ** 2:
                    state.drag_idx = i
                    return
            # Hit-test buttons
            if _hit_button(x, y, _BTN_RESET_CX, _BTN_CY):
                for sl, iv in zip(state.sliders, state.initial_values):
                    sl.value = iv
                state.edges_dirty = True
            elif _hit_button(x, y, _BTN_TAKE_CX, _BTN_CY):
                state.accepted = True
            elif _hit_button(x, y, _BTN_CANCEL_CX, _BTN_CY):
                state.cancelled = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drag_idx is not None:
            sl      = state.sliders[state.drag_idx]
            new_val = sl.value_from_x(x)
            if new_val != sl.value:
                sl.value          = new_val
                state.edges_dirty = True

    elif event == cv2.EVENT_LBUTTONUP:
        state.drag_idx = None


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
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]] | None:
    """
    Open a three-panel interactive window for Lab-based edge-map tuning.

    Layout (left → right):
      • Control panel  — custom-drawn sliders + Reset / Take / Cancel buttons
      • Edge panel     — inverted edge map with 10px white border
      • Master panel   — warped painting with 10px white border

    Returns
    -------
    (inverted_edges_full_res, (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi))
        or None if the user cancelled.
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

    # Pre-scale master once; add border; reuse every frame
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

    # Bootstrap frame so the HWND is real before maximize
    _bootstrap_ctrl = _draw_ctrl_panel(state.sliders)
    state.edge_panel  = _make_edge_panel(state.warped_display, state.sliders)
    state.edges_dirty = False
    _bt_h = max(_bootstrap_ctrl.shape[0],
                state.edge_panel.shape[0],
                state.master_panel.shape[0])
    _bootstrap_frame = np.hstack([
        _pad_to_height(_bootstrap_ctrl,    _bt_h),
        _pad_to_height(state.edge_panel,   _bt_h),
        _pad_to_height(state.master_panel, _bt_h),
    ])
    cv2.imshow(_EDGEMAP_WINDOW, _bootstrap_frame)
    cv2.waitKey(1)
    _maximize_window(_EDGEMAP_WINDOW)

    try:
        while True:
            if state.accepted or state.cancelled:
                break

            if state.edges_dirty:
                state.edge_panel  = _make_edge_panel(state.warped_display, state.sliders)
                state.edges_dirty = False

            ctrl_panel = _draw_ctrl_panel(state.sliders)

            target_h = max(
                ctrl_panel.shape[0],
                state.edge_panel.shape[0],
                state.master_panel.shape[0],
            )
            frame = np.hstack([
                _pad_to_height(ctrl_panel,         target_h),
                _pad_to_height(state.edge_panel,   target_h),
                _pad_to_height(state.master_panel, target_h),
            ])

            cv2.imshow(_EDGEMAP_WINDOW, frame)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("t"), ord("T")):
                state.accepted = True
            elif key in (ord("r"), ord("R")):
                for sl, iv in zip(state.sliders, state.initial_values):
                    sl.value = iv
                state.edges_dirty = True
            elif key in (ord("q"), ord("Q"), 27):
                state.cancelled = True

    finally:
        cv2.destroyWindow(_EDGEMAP_WINDOW)

    if not state.accepted:
        return None

    # Compute final edge map at full resolution, scaling blur proportionally
    # so the output matches what was seen in the UI panel (which was computed
    # at _DISPLAY_H pixels height with blur_ksize=5).
    final_vals   = tuple(sl.value for sl in state.sliders)
    full_edges = compute_lab_edges(state.warped_full, *final_vals)

    return full_edges, final_vals  # type: ignore[return-value]