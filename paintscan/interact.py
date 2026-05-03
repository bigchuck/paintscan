from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
_CTRL_W  = 300   # Lab control column
_CTRL2_W = 240   # info / Take-detail column
_BORDER  = 10

_CTRL_TOP  = 38
_BAND_H    = 85
_TRACK_X0  = 28
_TRACK_X1  = 272
_HANDLE_R  = 9

_BTN_W, _BTN_H = 72, 34
_BTN_CY        = _CTRL_TOP + 6 * _BAND_H + 35   # 583
_BTN_RESET_CX  = 50
_BTN_TAKE_CX   = 150
_BTN_DONE_CX   = 250

_TAKE_COUNT_Y  = _BTN_CY + _BTN_H // 2 + 18     # 618

_OC_BTN_W   = 116
_OC_BTN_H   = 28
_OC_BTN_CY  = 678
_BTN_OVL_CX = 75
_BTN_CLR_CX = 220

# Amber accent for local mode (BGR)
_AMBER: tuple = (0, 165, 255)

# Green accent for merge mode (BGR)
_MERGE_GREEN: tuple = (0, 180, 60)

# SEAL row (local mode only) — gap-close control sits below OVL/CLR
_SEAL_ROW_Y    = 714
_SEAL_MINUS_CX = 60
_SEAL_PLUS_CX  = 240
_SEAL_PBTN_W   = 38
_SEAL_PBTN_H   = 24
_SEAL_MIN      = 0
_SEAL_MAX      = 20
_SEAL_DEFAULT  = 4    # bridges gaps up to ~8 px wide; raise for loose edge maps

# Merge button row — sits below the SEAL row
_MERGE_BTN_CY  = 750
_MERGE_ADJ_PX  = 15   # dilation px for adjacency check — generous to handle thick Canny edges

# Print Preview button — sits below the merge row, global mode only
_BTN_PRINT_W   = 200
_BTN_PRINT_H   = 26
_BTN_PRINT_CY  = 784
_BTN_PRINT_CX  = _CTRL_W // 2   # 150

_TOTAL_H = 820   # ctrl panel height (extended to fit PRINT PREVIEW button)

# --- Filmstrip: two rows (Takes top, Color versions bottom) -----------------
_FILM_ROW_H   = 108   # height of each row
_FILM_H       = 216   # total strip height (two rows)
_FILM_SLOT_W  =  82
_FILM_THUMB_H =  76
_FILM_THUMB_W =  74
_FILM_START_X =   4
_FILM_BTN_W   =  64
_FILM_BTN_H   =  32
_FILM_BTN_CY  = _FILM_ROW_H // 2          # Take row button centre
_FILM_CLR_CY  = _FILM_ROW_H + _FILM_ROW_H // 2  # color row button centre
_FILM_LIVE_R  =  42
_FILM_SEED_R  = 116
_FILM_CLR_R   =  90   # COLORIZE button cx from right in color row

# Rose accent for colorize
_ROSE: tuple = (130, 90, 220)

# Colorize editor constants
_CLR_WINDOW   = "paintscan - colorize  (A=apply  S=save  R=reset  Esc=cancel)"
_CLR_SLIDER_DEFS: list[tuple] = [
    ("H target",   0, 179,  90, (180,  90, 255)),
    ("S target",   0, 255, 128, ( 60, 200,  60)),
    ("V scale %",  0, 200, 100, (200, 200,  60)),
]
_CLR_CTRL_TOP   = 38
_CLR_BAND_H     = 110
_CLR_BTN_CY     = _CLR_CTRL_TOP + 3 * _CLR_BAND_H + 48
_CLR_RESET_CX   = 75
_CLR_APPLY_CX   = 225
_CLR_SEAL_ROW_Y = _CLR_BTN_CY + 50
_CLR_SAVE_CY    = _CLR_SEAL_ROW_Y + 50
_CLR_CLR_CY     = _CLR_SAVE_CY + 44
_CLR_TOTAL_H    = _CLR_CLR_CY + 60

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

# Merge-mode slider label / handle colour variants (green family) — ⓜ = U+24DC
_MERGE_SLIDER_META: list[tuple] = [
    ("L* lo \u24dc", ( 50, 200,  50)),
    ("L* hi \u24dc", ( 70, 220,  70)),
    ("a* lo \u24dc", ( 50, 210, 150)),
    ("a* hi \u24dc", ( 70, 230, 170)),
    ("b* lo \u24dc", ( 50, 200, 210)),
    ("b* hi \u24dc", ( 70, 220, 230)),
]


# ---------------------------------------------------------------------------
# Per-Take record
# ---------------------------------------------------------------------------

@dataclass
class _TakeEntry:
    """Complete snapshot for one Take (or the auto-recorded Take 0).

    Fields
    ------
    index             Sequential Take number; 0 = auto-recorded initial state.
    edges_full        Full-resolution inv_gray for disk writing.  None for Take 0
                      and for takes reconstructed from a prior session (is_new=False)
                      since they were already written in the previous session.
    display_inv_gray  Display-resolution inv_gray used to render the preview panel
                      when this Take is selected in the filmstrip.
    global_thresholds (l_lo, l_hi, a_lo, a_hi, b_lo, b_hi) at Take time.
    local_info        None unless a local region was active when TAKE was pressed.
    seeded_from       Index of the Take whose thresholds seeded the current sliders,
                      or None if the user adjusted from the previous state freely.
    base_image        "master" for now; Phase 3 will add colour-version references.
    thumbnail         Small BGR image shown in the filmstrip strip.
    patches_snapshot  Serialisable copy of state.patches at Take time.
    is_new            True for takes created in this session; False for takes loaded
                      from a prior session (they don't need to be written to disk).
    """
    index:             int
    edges_full:        Optional[np.ndarray]
    display_inv_gray:  np.ndarray
    global_thresholds: tuple
    local_info:        Optional[dict]
    seeded_from:       Optional[int]
    base_image:        str
    thumbnail:         np.ndarray
    patches_snapshot:  list
    is_new:            bool = True
    color_versions:    list = field(default_factory=list)  # child color versions


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
    takes:          list = field(default_factory=list)   # list of _TakeEntry
    color_idx:      int  = 0
    overlay_on:     bool = False
    edges_dirty:    bool = True
    edge_panel:     Optional[np.ndarray] = None
    overlay_panel:  Optional[np.ndarray] = None
    # Cached raw inv_gray at display resolution — fed to the flood filler
    inv_gray_cache: Optional[np.ndarray] = None

    # Committed local patches
    patches:          list = field(default_factory=list)

    # Local region
    local_mode:       bool = False
    local_mask:       Optional[np.ndarray] = None
    local_bbox:       Optional[tuple] = None
    local_seed_disp:  Optional[tuple] = None
    local_seal:       int  = _SEAL_DEFAULT
    local_sliders:    list = field(default_factory=list)
    local_init_vals:  list = field(default_factory=list)
    local_drag_idx:   Optional[int] = None
    local_patch_idx:  Optional[int] = None
    local_dirty:      bool = False
    local_edge_panel: Optional[np.ndarray] = None
    local_zoom_panel: Optional[np.ndarray] = None

    # Take history / filmstrip
    # preview_take_idx: index of the Take shown in the middle panel.
    #   None means "live" — show the current working edge map.
    # seeded_from: the Take index whose thresholds seeded the current sliders.
    #   Recorded in the next new Take's metadata.
    preview_take_idx: Optional[int] = None
    seeded_from:      Optional[int] = None
    main_panel_h:     int = 0   # pixel height of the main three-panel row (set each frame)
    window_w:         int = 0   # total window pixel width (set each frame)

    # Phase 2 — super-areas and merge mode
    super_areas:        list = field(default_factory=list)   # list of SA dicts
    next_patch_id:      int  = 0    # monotonic; never reused
    next_super_area_id: int  = 0    # monotonic; never reused
    merge_mode:         bool = False
    merge_active_sa_id: Optional[int] = None   # SA whose sliders are shown
    merge_sliders:      list = field(default_factory=list)
    merge_init_vals:    list = field(default_factory=list)
    merge_drag_idx:     Optional[int] = None
    merge_dirty:        bool = False
    merge_edge_panel:   Optional[np.ndarray] = None

    # Phase 3 — colorize / filmstrip
    colorize_take_idx:      Optional[int] = None   # set when user hits COLORIZE
    print_preview_take_idx: Optional[int] = None   # set when user hits PRINT PREVIEW
    preview_color_ver_id:   Optional[int] = None   # which color swatch is selected
    pre_seed_values:        Optional[list] = None  # slider values before Take was selected
    has_take_zero:          bool = False            # Take-0 exists (user has pressed TAKE once)
    base_image:             str  = "master"        # base image label for this session


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
    patch_count: int = 0,
    local_patch_idx: int | None = None,
    merge_mode: bool = False,
    merge_sliders: list | None = None,
    merge_active_sa_id: int | None = None,
    super_area_count: int = 0,
    has_take_zero: bool = False,
    print_preview_enabled: bool = False,
) -> np.ndarray:
    panel = np.full((_TOTAL_H, _CTRL_W, 3), 28, dtype=np.uint8)

    # Title bar
    if merge_mode:
        if merge_active_sa_id is not None:
            title_bg   = (0, 120, 0)
            title_text = f"SUPER-AREA {merge_active_sa_id}"
        else:
            title_bg   = (0, 160, 60)
            title_text = "MERGE MODE"
    elif local_mode:
        if local_patch_idx is not None:
            title_bg   = (0, 80, 140)
            title_text = f"EDIT PATCH #{local_patch_idx + 1}"
        else:
            title_bg   = (0, 100, 180)
            title_text = "LOCAL MODE"
    elif not has_take_zero:
        title_bg   = (30, 30, 90)
        title_text = "Tune borders — press TAKE for T0"
    else:
        title_bg   = (45, 45, 45)
        title_text = "Lab Edge Thresholds"
    cv2.rectangle(panel, (0, 0), (_CTRL_W, _CTRL_TOP - 4), title_bg, -1)
    cv2.putText(panel, title_text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if merge_mode and merge_sliders and merge_active_sa_id is not None:
        active = merge_sliders
    elif local_mode and local_sliders:
        active = local_sliders
    else:
        active = sliders

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
    rst_col = _AMBER if local_mode else (_MERGE_GREEN if merge_mode else (90, 90, 30))
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

    # Patches count
    if patch_count > 0:
        ptch_text  = f"patches: {patch_count}"
        ptch_color = _AMBER
    else:
        ptch_text  = "patches: 0"
        ptch_color = (70, 70, 70)
    (tw, _), _ = cv2.getTextSize(ptch_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(panel, ptch_text, (_BTN_TAKE_CX - tw // 2, _TAKE_COUNT_Y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, ptch_color, 1, cv2.LINE_AA)

    # Super-area count
    if super_area_count > 0:
        sa_text  = f"super-areas: {super_area_count}"
        sa_color = _MERGE_GREEN
    else:
        sa_text  = "super-areas: 0"
        sa_color = (60, 60, 60)
    (tw, _), _ = cv2.getTextSize(sa_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    cv2.putText(panel, sa_text, (_BTN_TAKE_CX - tw // 2, _TAKE_COUNT_Y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, sa_color, 1, cv2.LINE_AA)

    # OVL / EXIT LOCAL / CLR row
    if local_mode:
        _draw_button(panel, "EXIT LOCAL", _BTN_OVL_CX, _OC_BTN_CY,
                     _AMBER, _OC_BTN_W, _OC_BTN_H)
    elif merge_mode:
        pass   # no OVL button in merge mode
    else:
        ovl_label = "OVL: ON " if overlay_on else "OVL: OFF"
        ovl_color = (35, 130, 35) if overlay_on else (60, 60, 60)
        _draw_button(panel, ovl_label, _BTN_OVL_CX, _OC_BTN_CY,
                     ovl_color, _OC_BTN_W, _OC_BTN_H)

    clr_label = f"CLR: {_EDGE_COLORS[color_idx][0]}"
    _draw_button(panel, clr_label, _BTN_CLR_CX, _OC_BTN_CY,
                 (55, 55, 90), _OC_BTN_W, _OC_BTN_H)

    # SEAL row: local controls / CLR THIS PATCH / CLR PTCH / hints
    if local_mode:
        if local_patch_idx is not None:
            _draw_button(panel, "CLR THIS PATCH", _CTRL_W // 2, _SEAL_ROW_Y,
                         (40, 40, 120), 160, _SEAL_PBTN_H)
        else:
            _draw_button(panel, "-", _SEAL_MINUS_CX, _SEAL_ROW_Y,
                         (60, 60, 80), _SEAL_PBTN_W, _SEAL_PBTN_H)
            _draw_button(panel, "+", _SEAL_PLUS_CX, _SEAL_ROW_Y,
                         (60, 60, 80), _SEAL_PBTN_W, _SEAL_PBTN_H)
            seal_text = f"SEAL: {local_seal}"
            (tw, _), _ = cv2.getTextSize(seal_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(panel, seal_text,
                        (_CTRL_W // 2 - tw // 2, _SEAL_ROW_Y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, _AMBER, 1, cv2.LINE_AA)
    elif patch_count > 0:
        _draw_button(panel, "CLR PTCH", _CTRL_W // 2, _SEAL_ROW_Y,
                     (60, 40, 40), 100, _SEAL_PBTN_H)
    elif not has_take_zero:
        hint = "Press TAKE to unlock local mode"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(panel, hint, (_CTRL_W // 2 - tw // 2, _SEAL_ROW_Y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 60, 60), 1, cv2.LINE_AA)
    else:
        hint = "Click edge panel to select area"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(panel, hint, (_CTRL_W // 2 - tw // 2, _SEAL_ROW_Y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    # MERGE / EXIT MERGE row
    if merge_mode:
        _draw_button(panel, "EXIT MERGE", _CTRL_W // 2, _MERGE_BTN_CY,
                     (0, 100, 20), 160, _BTN_H)
    elif patch_count >= 1 and has_take_zero:
        _draw_button(panel, "MERGE", _CTRL_W // 2, _MERGE_BTN_CY,
                     _MERGE_GREEN, 100, _BTN_H)
    else:
        hint2 = "Make patches to enable merge" if has_take_zero else "Press TAKE to unlock merge"
        col2  = (55, 55, 55) if has_take_zero else (70, 50, 50)
        (tw, _), _ = cv2.getTextSize(hint2, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(panel, hint2, (_CTRL_W // 2 - tw // 2, _MERGE_BTN_CY + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col2, 1, cv2.LINE_AA)

    # PRINT PREVIEW button — global mode only; dims when no Take is selected
    if not local_mode and not merge_mode:
        pp_col = (85, 55, 125) if print_preview_enabled else (42, 38, 50)
        _draw_button(panel, "PRINT PREVIEW", _BTN_PRINT_CX, _BTN_PRINT_CY,
                     pp_col, _BTN_PRINT_W, _BTN_PRINT_H)

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


def _compute_composite_inv_gray(
    warped_display: np.ndarray,
    sliders: list,
    patches: list,
    super_areas: list | None = None,
) -> np.ndarray:
    """
    Compute a merged inv_gray from global thresholds plus all committed patches.
    Patches that belong to a super-area use the super-area's thresholds instead
    of their own.  inv_gray convention: 255 = background, 0 = edge.
    """
    g_vals    = [sl.value for sl in sliders]
    composite = compute_lab_edges(warped_display, *g_vals)
    sa_list   = super_areas or []
    for patch in patches:
        sa_id  = patch.get("super_area_id")
        if sa_id is not None:
            sa = next((s for s in sa_list if s["super_area_id"] == sa_id), None)
            thresh = sa["thresholds"] if sa else patch["thresholds"]
        else:
            thresh = patch["thresholds"]
        p_inv = compute_lab_edges(warped_display, *thresh)
        mask  = patch["mask"]
        composite[mask > 0] = np.minimum(composite[mask > 0], p_inv[mask > 0])
    return composite


def _render_inv_gray(inv_gray: np.ndarray, color_idx: int) -> np.ndarray:
    """Colorise an inv_gray map and add border — does not recompute edges."""
    _name, edge_bgr, bg_bgr = _EDGE_COLORS[color_idx]
    out = np.full((*inv_gray.shape, 3), bg_bgr, dtype=np.uint8)
    out[inv_gray == 0] = edge_bgr
    return _add_border(out, _BORDER)


def _make_edge_panel(
    warped_display: np.ndarray,
    sliders: list,
    color_idx: int = 0,
    patches: list | None = None,
    super_areas: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bordered_BGR_panel, composite_inv_gray_cache)."""
    inv_gray = _compute_composite_inv_gray(warped_display, sliders, patches or [], super_areas)
    return _render_inv_gray(inv_gray, color_idx), inv_gray


def _make_overlay_panel(warped_display, sliders, color_idx, patches=None, super_areas=None):
    inv_gray = _compute_composite_inv_gray(warped_display, sliders, patches or [], super_areas)
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


def _scale_to_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Scale *image* so its longer side equals *max_dim*, preserving aspect ratio."""
    h, w   = image.shape[:2]
    larger = max(h, w)
    if larger == max_dim:
        return image.copy()
    scale  = max_dim / larger
    new_w  = max(1, int(round(w * scale)))
    new_h  = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary display in pixels."""
    try:
        import ctypes
        u32 = ctypes.windll.user32
        return int(u32.GetSystemMetrics(0)), int(u32.GetSystemMetrics(1))
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        pass
    return 1920, 1080   # safe fallback


def _compute_panel_size(image: np.ndarray) -> tuple[int, int]:
    """Return the (width, height) each display panel image should be rendered at."""
    ih, iw     = image.shape[:2]
    sw, sh     = _get_screen_size()
    os_chrome  = 72

    avail_w = (sw - _CTRL_W - _CTRL2_W - _BORDER * 8) // 2
    avail_h = sh - _FILM_H - _BORDER * 4 - os_chrome

    scale  = min(avail_w / max(iw, 1), avail_h / max(ih, 1))
    new_w  = max(1, int(iw * scale))
    new_h  = max(1, int(ih * scale))
    return new_w, new_h


def _compute_panel_size_colorize(image: np.ndarray) -> tuple[int, int]:
    """Panel size for the 3-column colorize window (ctrl + edge + painted). No filmstrip."""
    ih, iw    = image.shape[:2]
    sw, sh    = _get_screen_size()
    os_chrome = 72
    avail_w   = (sw - _CTRL_W - _BORDER * 6) // 2
    avail_h   = sh - _BORDER * 4 - os_chrome
    scale     = min(avail_w / max(iw, 1), avail_h / max(ih, 1))
    return max(1, int(iw * scale)), max(1, int(ih * scale))


def _pad_to_height(img: np.ndarray, h: int) -> np.ndarray:
    dh = h - img.shape[0]
    if dh <= 0:
        return img
    return np.vstack([img, np.full((dh, img.shape[1], 3), 28, dtype=np.uint8)])


# ---------------------------------------------------------------------------
# Filmstrip helpers
# ---------------------------------------------------------------------------

def _generate_thumbnail(inv_gray: np.ndarray) -> np.ndarray:
    """Scale *inv_gray* to _FILM_THUMB_H, letterbox to _FILM_THUMB_W, return BGR."""
    h, w   = inv_gray.shape
    scale  = _FILM_THUMB_H / max(1, h)
    new_w  = max(1, int(round(w * scale)))
    small  = cv2.resize(inv_gray, (new_w, _FILM_THUMB_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((_FILM_THUMB_H, _FILM_THUMB_W), 200, dtype=np.uint8)
    x0     = max(0, (_FILM_THUMB_W - new_w) // 2)
    x1     = min(_FILM_THUMB_W, x0 + new_w)
    canvas[:, x0:x1] = small[:, : x1 - x0]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _generate_color_thumbnail(bgr: np.ndarray) -> np.ndarray:
    """Scale a BGR color image to filmstrip thumbnail size."""
    h, w   = bgr.shape[:2]
    scale  = _FILM_THUMB_H / max(1, h)
    new_w  = max(1, int(round(w * scale)))
    small  = cv2.resize(bgr, (new_w, _FILM_THUMB_H), interpolation=cv2.INTER_AREA)
    canvas = np.full((_FILM_THUMB_H, _FILM_THUMB_W, 3), 50, dtype=np.uint8)
    x0 = max(0, (_FILM_THUMB_W - new_w) // 2)
    x1 = min(_FILM_THUMB_W, x0 + new_w)
    canvas[:, x0:x1] = small[:, :x1 - x0]
    return canvas


# ---------------------------------------------------------------------------
# Local region helpers
# ---------------------------------------------------------------------------

def _flood_fill_region(
    inv_gray: np.ndarray,
    seed_x: int,
    seed_y: int,
    seal_px: int = 0,
) -> np.ndarray:
    """Flood-fill from (seed_x, seed_y) through connected white pixels in inv_gray."""
    h, w = inv_gray.shape

    if seal_px > 0:
        k    = 2 * seal_px + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges_fg  = cv2.bitwise_not(inv_gray)
        edges_fat = cv2.dilate(edges_fg, kern)
        work_img  = cv2.bitwise_not(edges_fat)
    else:
        work_img = inv_gray

    sx, sy = seed_x, seed_y

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
    patches: list | None = None,
    super_areas: list | None = None,
) -> np.ndarray:
    """Composite edge panel for local mode."""
    composite_base = _compute_composite_inv_gray(
        warped_display, global_sliders, patches or [], super_areas)

    l_vals    = [sl.value for sl in local_sliders]
    local_inv = compute_lab_edges(warped_display, *l_vals)

    merged = composite_base.copy()
    merged[local_mask > 0] = local_inv[local_mask > 0]

    _name, edge_bgr, bg_bgr = _EDGE_COLORS[color_idx]
    out = np.full((*merged.shape, 3), bg_bgr, dtype=np.uint8)
    out[merged == 0] = edge_bgr

    outside = local_mask == 0
    out[outside] = (out[outside].astype(np.float32) * 0.35).astype(np.uint8)

    kernel  = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(local_mask, kernel, iterations=1)
    ring    = (dilated > 0) & (local_mask == 0)
    out[ring] = _AMBER

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
    x0, y0, x1, y1 = local_bbox
    h, w = warped_display.shape[:2]
    pad_x = max(10, int((x1 - x0) * 0.15))
    pad_y = max(10, int((y1 - y0) * 0.15))
    cx0 = max(0, x0 - pad_x);  cy0 = max(0, y0 - pad_y)
    cx1 = min(w, x1 + pad_x);  cy1 = min(h, y1 + pad_y)
    crop   = warped_display[cy0:cy1, cx0:cx1].copy()
    zoomed = _scale_to_height(crop, h)
    cv2.rectangle(zoomed, (0, 0), (64, 22), _AMBER, -1)
    cv2.putText(zoomed, "ZOOM", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return _add_border(zoomed, _BORDER, color=_AMBER)


# ---------------------------------------------------------------------------
# Local mode transitions
# ---------------------------------------------------------------------------

def _make_local_sliders_from_thresholds(thresholds: tuple) -> list:
    return [
        _Slider(
            label   = _LOCAL_SLIDER_META[i][0],
            min_val = defn[1],
            max_val = defn[2],
            value   = max(defn[1], min(defn[2], thresholds[i])),
            color   = _LOCAL_SLIDER_META[i][1],
        )
        for i, defn in enumerate(_SLIDER_DEFS)
    ]


def _enter_local_mode(state: _EdgemapState, disp_x: int, disp_y: int) -> None:
    if not state.has_take_zero:
        return
    for i in range(len(state.patches) - 1, -1, -1):
        patch = state.patches[i]
        if patch["mask"][disp_y, disp_x] > 0:
            _enter_patch_edit_mode(state, i)
            return

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
    state.local_patch_idx = None
    state.local_mode      = True
    state.local_dirty     = True


def _enter_patch_edit_mode(state: _EdgemapState, patch_idx: int) -> None:
    patch = state.patches[patch_idx]
    state.local_mask      = patch["mask"].copy()
    state.local_bbox      = patch["bbox"]
    state.local_seed_disp = None
    state.local_sliders   = _make_local_sliders_from_thresholds(patch["thresholds"])
    state.local_init_vals = list(patch["thresholds"])
    state.local_patch_idx = patch_idx
    state.local_mode      = True
    state.local_dirty     = True


def _rerun_flood_fill(state: _EdgemapState) -> None:
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
    state.local_mode       = False
    state.local_mask       = None
    state.local_bbox       = None
    state.local_seed_disp  = None
    state.local_patch_idx  = None
    state.local_sliders    = []
    state.local_init_vals  = []
    state.local_drag_idx   = None
    state.local_edge_panel = None
    state.local_zoom_panel = None


def _commit_local_patch(state: _EdgemapState) -> None:
    if state.local_mask is None or not state.local_sliders:
        return
    local_vals  = tuple(sl.value for sl in state.local_sliders)
    global_vals = tuple(sl.value for sl in state.sliders)

    if state.local_patch_idx is not None:
        original_vals = tuple(state.local_init_vals)
        if local_vals == original_vals:
            return
    else:
        if local_vals == global_vals:
            return

    h, w = state.warped_display.shape[:2]
    if state.local_patch_idx is not None:
        orig = state.patches[state.local_patch_idx]
        new_patch: dict = {
            "patch_id":      orig["patch_id"],
            "super_area_id": orig.get("super_area_id"),
            "mask":          state.local_mask.copy(),
            "thresholds":    local_vals,
            "bbox":          state.local_bbox,
            "seed_norm":     orig.get("seed_norm"),
            "seal":          orig.get("seal", _SEAL_DEFAULT),
        }
        state.patches[state.local_patch_idx] = new_patch
    else:
        new_patch = {
            "patch_id":      _alloc_patch_id(state),
            "super_area_id": None,
            "mask":          state.local_mask.copy(),
            "thresholds":    local_vals,
            "bbox":          state.local_bbox,
        }
        if state.local_seed_disp is not None:
            sx, sy = state.local_seed_disp
            new_patch["seed_norm"] = (sx / max(1, w), sy / max(1, h))
            new_patch["seal"]      = state.local_seal
        state.patches.append(new_patch)


def _patches_to_session_data(patches: list) -> list:
    """Convert state.patches to a JSON-serialisable list (no numpy arrays)."""
    out = []
    for p in patches:
        sn = p.get("seed_norm")
        if sn is None:
            continue
        out.append({
            "patch_id":      p.get("patch_id", 0),
            "super_area_id": p.get("super_area_id"),
            "seed_norm":     [round(sn[0], 6), round(sn[1], 6)],
            "seal":          int(p.get("seal", _SEAL_DEFAULT)),
            "thresholds":    list(p["thresholds"]),
        })
    return out


def _restore_patches_from_session(
    warped_display: np.ndarray,
    patches_data: list,
    global_sliders: list,
) -> tuple[list, int]:
    """Reconstruct patch masks from saved session data."""
    if not patches_data:
        return [], 0

    g_vals   = [sl.value for sl in global_sliders]
    inv_gray = compute_lab_edges(warped_display, *g_vals)
    h, w     = warped_display.shape[:2]
    restored = []
    max_pid  = -1

    for pd in patches_data:
        try:
            nx, ny    = pd["seed_norm"]
            seal      = int(pd.get("seal", _SEAL_DEFAULT))
            thresh    = tuple(pd["thresholds"])
            pid       = int(pd.get("patch_id", 0))
            sa_id     = pd.get("super_area_id")
            sx        = max(0, min(int(round(nx * w)), w - 1))
            sy        = max(0, min(int(round(ny * h)), h - 1))
            mask      = _flood_fill_region(inv_gray, sx, sy, seal_px=seal)
            bbox      = _bbox_from_mask(mask)
            if bbox is None:
                continue
            max_pid = max(max_pid, pid)
            restored.append({
                "patch_id":      pid,
                "super_area_id": sa_id,
                "mask":          mask,
                "thresholds":    thresh,
                "bbox":          bbox,
                "seed_norm":     (nx, ny),
                "seal":          seal,
            })
        except Exception:
            continue

    return restored, max_pid + 1


# ---------------------------------------------------------------------------
# Phase 2 — super-area helpers
# ---------------------------------------------------------------------------

def _alloc_patch_id(state: _EdgemapState) -> int:
    pid = state.next_patch_id
    state.next_patch_id += 1
    return pid


def _alloc_super_area_id(state: _EdgemapState) -> int:
    sid = state.next_super_area_id
    state.next_super_area_id += 1
    return sid


def _get_sa(state: _EdgemapState, sa_id: int) -> dict | None:
    return next((s for s in state.super_areas if s["super_area_id"] == sa_id), None)


def _masks_adjacent(mask_a: np.ndarray, mask_b: np.ndarray, px: int = _MERGE_ADJ_PX) -> bool:
    """Return True if mask_a and mask_b are within px pixels of each other."""
    k = 2 * px + 1
    kern    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate((mask_a > 0).astype(np.uint8), kern)
    return bool(np.any(dilated & (mask_b > 0).astype(np.uint8)))


def _make_merge_sliders_from_thresholds(thresholds: tuple) -> list:
    return [
        _Slider(
            label   = _MERGE_SLIDER_META[i][0],
            min_val = defn[1],
            max_val = defn[2],
            value   = max(defn[1], min(defn[2], thresholds[i])),
            color   = _MERGE_SLIDER_META[i][1],
        )
        for i, defn in enumerate(_SLIDER_DEFS)
    ]


def _enter_sa_edit(state: _EdgemapState, sa_id: int) -> None:
    """Open super-area Lab slider editing for sa_id."""
    sa = _get_sa(state, sa_id)
    if sa is None:
        return
    state.merge_active_sa_id = sa_id
    state.merge_sliders      = _make_merge_sliders_from_thresholds(sa["thresholds"])
    state.merge_init_vals    = list(sa["thresholds"])
    state.merge_drag_idx     = None
    state.merge_dirty        = True


def _exit_sa_edit(state: _EdgemapState) -> None:
    """Commit current SA sliders and leave SA edit (stay in merge mode)."""
    _commit_sa_edit(state)
    state.merge_active_sa_id = None
    state.merge_sliders      = []
    state.merge_init_vals    = []
    state.merge_drag_idx     = None


def _commit_sa_edit(state: _EdgemapState) -> None:
    """Write current merge slider values back to the active super-area."""
    if state.merge_active_sa_id is None or not state.merge_sliders:
        return
    sa = _get_sa(state, state.merge_active_sa_id)
    if sa is None:
        return
    sa["thresholds"] = tuple(sl.value for sl in state.merge_sliders)
    state.edges_dirty = True


def _unmerge_patch(state: _EdgemapState, patch_id: int) -> None:
    """Remove a patch from its super-area.  Dissolve SA if only 1 member remains."""
    patch = next((p for p in state.patches if p["patch_id"] == patch_id), None)
    if patch is None:
        return
    sa_id = patch.get("super_area_id")
    if sa_id is None:
        return
    sa = _get_sa(state, sa_id)
    if sa and patch_id in sa["patch_ids"]:
        sa["patch_ids"].remove(patch_id)
        if len(sa["patch_ids"]) <= 1:
            for p in state.patches:
                if p.get("super_area_id") == sa_id:
                    p["super_area_id"] = None
            state.super_areas = [s for s in state.super_areas if s["super_area_id"] != sa_id]
            if state.merge_active_sa_id == sa_id:
                state.merge_active_sa_id = None
                state.merge_sliders      = []
                state.merge_init_vals    = []
    patch["super_area_id"] = None
    state.edges_dirty  = True
    state.merge_dirty  = True


def _unpatch_by_id(state: _EdgemapState, patch_id: int) -> None:
    """Delete a patch entirely.  Unmerges from SA first if necessary."""
    _unmerge_patch(state, patch_id)
    state.patches     = [p for p in state.patches if p["patch_id"] != patch_id]
    state.edges_dirty = True
    state.merge_dirty = True


def _do_merge_lclick(state: _EdgemapState, disp_x: int, disp_y: int) -> None:
    """Handle a left-click on the edge panel while in merge mode."""
    try:
        _do_merge_lclick_inner(state, disp_x, disp_y)
    except Exception as exc:
        print(f"[MERGE lclick error] {exc!r}")


def _do_merge_lclick_inner(state: _EdgemapState, disp_x: int, disp_y: int) -> None:
    for patch in reversed(state.patches):
        if patch["mask"][disp_y, disp_x] > 0:
            sa_id = patch.get("super_area_id")
            if sa_id is not None:
                _enter_sa_edit(state, sa_id)
                state.merge_dirty = True
            return

    if state.inv_gray_cache is None:
        return
    mask = _flood_fill_region(state.inv_gray_cache, disp_x, disp_y,
                               seal_px=state.local_seal)
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return

    adjacent = None
    for patch in reversed(state.patches):
        if _masks_adjacent(mask, patch["mask"]):
            adjacent = patch
            break

    if adjacent is None:
        print("[MERGE] No adjacent patch found — click closer to an existing patch")
        return

    new_pid = _alloc_patch_id(state)
    h, w    = state.warped_display.shape[:2]
    new_patch: dict = {
        "patch_id":      new_pid,
        "super_area_id": None,
        "mask":          mask,
        "thresholds":    tuple(sl.value for sl in state.sliders),
        "bbox":          bbox,
        "seed_norm":     (disp_x / max(1, w), disp_y / max(1, h)),
        "seal":          state.local_seal,
    }

    adj_sa_id = adjacent.get("super_area_id")
    if adj_sa_id is None:
        new_sa_id = _alloc_super_area_id(state)
        adjacent["super_area_id"] = new_sa_id
        new_patch["super_area_id"] = new_sa_id
        sa = {
            "super_area_id": new_sa_id,
            "thresholds":    adjacent["thresholds"],
            "patch_ids":     [adjacent["patch_id"], new_pid],
        }
        state.super_areas.append(sa)
        state.patches.append(new_patch)
        _enter_sa_edit(state, new_sa_id)
    else:
        sa = _get_sa(state, adj_sa_id)
        new_patch["super_area_id"] = adj_sa_id
        sa["patch_ids"].append(new_pid)
        state.patches.append(new_patch)
        _enter_sa_edit(state, adj_sa_id)

    state.edges_dirty = True
    state.merge_dirty = True


def _do_merge_rclick(state: _EdgemapState, disp_x: int, disp_y: int) -> None:
    """Handle a right-click on the edge panel while in merge mode."""
    try:
        for patch in reversed(state.patches):
            if patch["mask"][disp_y, disp_x] > 0:
                sa_id = patch.get("super_area_id")
                if sa_id is not None:
                    _unmerge_patch(state, patch["patch_id"])
                else:
                    _unpatch_by_id(state, patch["patch_id"])
                return
    except Exception as exc:
        print(f"[MERGE rclick error] {exc!r}")


def _make_merge_edge_panel(
    warped_display: np.ndarray,
    sliders: list,
    patches: list,
    super_areas: list,
    color_idx: int,
    active_sa_id: int | None = None,
) -> np.ndarray:
    """Edge panel for merge mode: base composite + coloured rings per patch/SA."""
    composite = _compute_composite_inv_gray(warped_display, sliders, patches, super_areas)
    _name, edge_bgr, bg_bgr = _EDGE_COLORS[color_idx]
    out = np.full((*composite.shape, 3), bg_bgr, dtype=np.uint8)
    out[composite == 0] = edge_bgr

    kern = np.ones((5, 5), np.uint8)
    for patch in patches:
        sa_id   = patch.get("super_area_id")
        dilated = cv2.dilate(patch["mask"], kern, iterations=2)
        ring    = (dilated > 0) & (patch["mask"] == 0)
        if sa_id is None:
            ring_color = _AMBER
        elif sa_id == active_sa_id:
            ring_color = (80, 255, 80)
        else:
            ring_color = _MERGE_GREEN
        out[ring] = ring_color

    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 140, 40), -1)
    msg = "MERGE  \u2014  L-click=join/edit SA  R-click=unmerge/unpatch  Esc=exit"
    cv2.putText(out, msg, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return _add_border(out, _BORDER, color=_MERGE_GREEN)


def _super_areas_to_session_data(super_areas: list) -> list:
    return [
        {
            "super_area_id": sa["super_area_id"],
            "thresholds":    list(sa["thresholds"]),
            "patch_ids":     list(sa["patch_ids"]),
        }
        for sa in super_areas
    ]


def _restore_super_areas_from_session(super_areas_data: list) -> list:
    result = []
    for sd in super_areas_data:
        try:
            result.append({
                "super_area_id": int(sd["super_area_id"]),
                "thresholds":    tuple(sd["thresholds"]),
                "patch_ids":     list(sd["patch_ids"]),
            })
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Filmstrip — rendering and interaction
# ---------------------------------------------------------------------------

def _draw_info_panel(state: "_EdgemapState", panel_h: int) -> np.ndarray:
    """Secondary column: Take details, Lab bars, color swatches."""
    W = _CTRL2_W
    P = np.full((panel_h, W, 3), 22, dtype=np.uint8)

    def lbl(txt, y, col=(140,140,140), sc=0.40):
        cv2.putText(P, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    def hline(y):
        cv2.line(P, (6, y), (W-6, y), (50,50,50), 1)

    if not state.has_take_zero:
        cv2.rectangle(P, (0,0), (W,24), (30,30,90), -1)
        cv2.putText(P, "No T0 yet", (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180,180,255), 1, cv2.LINE_AA)
        my = 44
        for line in ["Tune Lab sliders,", "then press TAKE", "to set T0.", "",
                     "Local, merge,", "colorize unlock", "after T0."]:
            lbl(line, my, col=(100,100,160)); my += 18
        return P

    sel = state.preview_take_idx
    entry = next((t for t in state.takes if t.index == sel), None) if sel is not None else None
    hdr = "LIVE" if sel is None else ("T0" if sel == 0 else f"T{sel}")
    hdr_col = (35,130,35) if sel is None else _AMBER
    thresh = (entry.global_thresholds if entry else tuple(sl.value for sl in state.sliders))

    cv2.rectangle(P, (0,0), (W,24), hdr_col, -1)
    cv2.putText(P, hdr, (8,17), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
    if entry and entry.seeded_from is not None:
        sf = f"\u2192T{entry.seeded_from}"
        (tw,_),_ = cv2.getTextSize(sf, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(P, sf, (W-tw-6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1, cv2.LINE_AA)
    y = 30

    hline(y); y += 6
    for i,(nm,col) in enumerate(zip(("L*","a*","b*"),((180,180,180),(80,230,100),(50,230,240)))):
        lo = thresh[i*2]; hi = thresh[i*2+1]
        lbl(f"{nm} {lo}-{hi}", y+11, col=col)
        bx0,bx1 = W-68, W-8
        cv2.line(P,(bx0,y+7),(bx1,y+7),(50,50,50),6)
        p0 = bx0+int(lo/255*(bx1-bx0)); p1 = bx0+int(hi/255*(bx1-bx0))
        cv2.line(P,(p0,y+7),(p1,y+7),col,4)
        y += 20
    y += 4

    hline(y); y += 4
    lbl(f"patches: {len(state.patches)}", y+12, col=_AMBER if state.patches else (70,70,70))
    lbl(f"super-areas: {len(state.super_areas)}", y+26, col=_MERGE_GREEN if state.super_areas else (60,60,60))
    y += 36

    if entry is not None and entry.color_versions:
        hline(y); y += 4
        cv2.rectangle(P,(0,y),(W,y+20),(60,30,70),-1)
        cv2.putText(P,"COLOR VERSIONS",(6,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1,cv2.LINE_AA)
        y += 22
        TH, TW2 = 38, 54
        for cv_rec in entry.color_versions:
            if y+TH+8 > panel_h-4:
                lbl("+more", y+12, col=(80,80,80)); break
            vid = cv_rec.get("version_id","?")
            is_sel = (state.preview_color_ver_id == vid)
            bdr = _ROSE if is_sel else (60,40,60)
            thumb = cv_rec.get("thumbnail")
            if thumb is not None:
                try:
                    th_s = cv2.resize(thumb,(TW2,TH),interpolation=cv2.INTER_AREA)
                    P[y:y+TH, 8:8+TW2] = th_s
                    cv2.rectangle(P,(7,y-1),(8+TW2,y+TH),bdr,1)
                except Exception:
                    pass
            lbl(f"C{vid}", y+13, col=_ROSE if is_sel else (160,120,180), sc=0.42)
            ts = cv_rec.get("timestamp","")[:10]
            if ts: lbl(ts, y+27, col=(80,80,80), sc=0.34)
            y += TH+8
    return P


def _make_filmstrip_panel(state: "_EdgemapState", total_w: int) -> np.ndarray:
    """Two-row filmstrip: Takes (top) + Color versions (bottom)."""
    panel  = np.full((_FILM_H, total_w, 3), 18, dtype=np.uint8)
    dimmed = state.local_mode or state.merge_mode

    cv2.line(panel, (0,0),           (total_w,0),           (55,55,55), 1)
    cv2.line(panel, (0,_FILM_ROW_H), (total_w,_FILM_ROW_H), (40,40,40), 1)

    ty0    = (_FILM_ROW_H - _FILM_THUMB_H - 16) // 2
    ty1    = _FILM_ROW_H + ty0
    btn_rx = total_w - _FILM_SEED_R - _FILM_BTN_W//2 - 12

    # Row 0: Takes
    for i, entry in enumerate(state.takes):
        sx = _FILM_START_X + i*_FILM_SLOT_W
        tx = sx + (_FILM_SLOT_W - _FILM_THUMB_W) // 2
        if tx+_FILM_THUMB_W > btn_rx:
            cv2.putText(panel, f"+{len(state.takes)-i} more",
                        (btn_rx-60, _FILM_ROW_H//2+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA); break
        sel  = (not dimmed) and (state.preview_take_idx == entry.index)
        cv2.rectangle(panel,(tx-2,ty0-2),(tx+_FILM_THUMB_W+1,ty0+_FILM_THUMB_H+1),
                      _AMBER if sel else (50,50,50), 2 if sel else 1)
        roi = panel[ty0:ty0+_FILM_THUMB_H, tx:tx+_FILM_THUMB_W]
        src = entry.thumbnail[:_FILM_THUMB_H, :_FILM_THUMB_W]
        roi[:] = (src.astype(np.float32)*0.35).astype(np.uint8) if dimmed else src
        lbl2 = "T0" if entry.index==0 else f"T{entry.index}"
        lc   = _AMBER if sel else ((60,60,60) if dimmed else (130,130,130))
        (lw,_),_ = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(panel, lbl2, (tx+(_FILM_THUMB_W-lw)//2, ty0+_FILM_THUMB_H+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1, cv2.LINE_AA)
        if entry.seeded_from is not None and not dimmed:
            sf = f"\u2192T{entry.seeded_from}"
            (sw2,_),_ = cv2.getTextSize(sf, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
            cv2.putText(panel, sf, (tx+(_FILM_THUMB_W-sw2)//2, ty0+_FILM_THUMB_H+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,100,120), 1, cv2.LINE_AA)

    if dimmed:
        msg = "Exit merge/local mode to navigate Takes"
        (hw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(panel, msg, (total_w//2-hw//2, _FILM_ROW_H//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (55,55,55), 1, cv2.LINE_AA)
        return panel

    # LIVE / SEED buttons
    live_cx = total_w - _FILM_LIVE_R
    _draw_button(panel, "LIVE", live_cx, _FILM_BTN_CY,
                 (35,130,35) if state.preview_take_idx is None else (50,50,50),
                 _FILM_BTN_W, _FILM_BTN_H)
    if state.preview_take_idx is not None:
        seed_cx = total_w - _FILM_SEED_R
        _draw_button(panel, "SEED", seed_cx, _FILM_BTN_CY,
                     (0,100,180), _FILM_BTN_W, _FILM_BTN_H)

    # Row 1: Color versions for selected Take
    sel_idx   = state.preview_take_idx
    sel_entry = next((t for t in state.takes if t.index == sel_idx), None) if sel_idx is not None else None
    if sel_entry is not None:
        cvs = sel_entry.color_versions
        for j, cv_rec in enumerate(cvs):
            sx2 = _FILM_START_X + j*_FILM_SLOT_W
            tx2 = sx2 + (_FILM_SLOT_W - _FILM_THUMB_W) // 2
            if tx2+_FILM_THUMB_W > btn_rx:
                cv2.putText(panel, f"+{len(cvs)-j} more",
                            (btn_rx-60, _FILM_ROW_H+_FILM_ROW_H//2+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,80,80), 1, cv2.LINE_AA); break
            vid = cv_rec.get("version_id", j)
            cv_sel = (state.preview_color_ver_id == vid)
            cv2.rectangle(panel,(tx2-2,ty1-2),(tx2+_FILM_THUMB_W+1,ty1+_FILM_THUMB_H+1),
                          _ROSE if cv_sel else (60,40,60), 2 if cv_sel else 1)
            thumb = cv_rec.get("thumbnail")
            if thumb is not None:
                try:
                    panel[ty1:ty1+_FILM_THUMB_H, tx2:tx2+_FILM_THUMB_W] = \
                        thumb[:_FILM_THUMB_H, :_FILM_THUMB_W]
                except Exception:
                    pass
            (lw,_),_ = cv2.getTextSize(f"C{vid}", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(panel, f"C{vid}",
                        (tx2+(_FILM_THUMB_W-lw)//2, ty1+_FILM_THUMB_H+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        _ROSE if cv_sel else (100,70,120), 1, cv2.LINE_AA)
        # COLORIZE button
        clr_cx = total_w - _FILM_CLR_R
        clr_col = _ROSE if state.has_take_zero else (60,40,60)
        _draw_button(panel, "COLORIZE", clr_cx, _FILM_CLR_CY, clr_col, 88, _FILM_BTN_H)
    else:
        hint3 = "Select a Take to colorize"
        (hw,_),_ = cv2.getTextSize(hint3, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(panel, hint3, (total_w//2-hw//2, _FILM_ROW_H+_FILM_ROW_H//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (55,55,55), 1, cv2.LINE_AA)

    return panel


def _seed_from_take(state: "_EdgemapState", take_idx: int) -> None:
    entry = next((t for t in state.takes if t.index == take_idx), None)
    if entry is None:
        return
    for sl, val in zip(state.sliders, entry.global_thresholds):
        sl.value = val
    state.seeded_from = take_idx
    state.edges_dirty = True


def _handle_filmstrip_click(state: "_EdgemapState", x: int, raw_y: int) -> None:
    """raw_y is relative to filmstrip top."""
    fy_bot = raw_y - _FILM_ROW_H

    # Top row: Take thumbnails + LIVE/SEED buttons
    if raw_y < _FILM_ROW_H:
        btn_rx = state.window_w - _FILM_SEED_R - _FILM_BTN_W//2 - 12 if state.window_w > 0 else 9999

        # LIVE button
        if state.window_w > 0:
            live_cx = state.window_w - _FILM_LIVE_R
            if _hit_button(x, raw_y, live_cx, _FILM_BTN_CY, _FILM_BTN_W, _FILM_BTN_H):
                if state.pre_seed_values is not None:
                    for sl, v in zip(state.sliders, state.pre_seed_values):
                        sl.value = v
                    state.pre_seed_values = None
                state.preview_take_idx     = None
                state.preview_color_ver_id = None
                state.edges_dirty          = True
                return

            # SEED button (only when a Take is previewed)
            if state.preview_take_idx is not None:
                seed_cx = state.window_w - _FILM_SEED_R
                if _hit_button(x, raw_y, seed_cx, _FILM_BTN_CY, _FILM_BTN_W, _FILM_BTN_H):
                    _seed_from_take(state, state.preview_take_idx)
                    return

        # Take thumbnails
        if x >= _FILM_START_X:
            slot = (x - _FILM_START_X) // _FILM_SLOT_W
            if 0 <= slot < len(state.takes):
                entry = state.takes[slot]
                tx = _FILM_START_X + slot*_FILM_SLOT_W + (_FILM_SLOT_W - _FILM_THUMB_W)//2
                if tx + _FILM_THUMB_W > btn_rx:
                    return   # truncated slot
                if state.preview_take_idx == entry.index:
                    # Deselect — restore pre-seed values
                    if state.pre_seed_values is not None:
                        for sl, v in zip(state.sliders, state.pre_seed_values):
                            sl.value = v
                        state.pre_seed_values = None
                    state.preview_take_idx     = None
                    state.preview_color_ver_id = None
                    state.edges_dirty          = True
                else:
                    # Select this Take
                    state.pre_seed_values      = [sl.value for sl in state.sliders]
                    state.preview_take_idx     = entry.index
                    state.preview_color_ver_id = None
                    _seed_from_take(state, entry.index)
        return

    # Bottom row
    if not state.has_take_zero or state.preview_take_idx is None:
        return
    sel_entry = next((t for t in state.takes if t.index == state.preview_take_idx), None)
    if sel_entry is None:
        return
    # COLORIZE button
    if state.window_w > 0:
        clr_cx = state.window_w - _FILM_CLR_R
        if _hit_button(x, fy_bot, clr_cx, _FILM_BTN_CY, 88, _FILM_BTN_H):
            state.colorize_take_idx = state.preview_take_idx
            state.done = True
            return
    # Color version thumbnail
    if x >= _FILM_START_X:
        slot = (x - _FILM_START_X) // _FILM_SLOT_W
        cvs  = sel_entry.color_versions
        if 0 <= slot < len(cvs):
            vid = cvs[slot].get("version_id")
            state.preview_color_ver_id = None if state.preview_color_ver_id == vid else vid


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def _edgemap_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    state: _EdgemapState = param
    ep_w = state.warped_display.shape[1] + 2 * _BORDER
    _EP_X  = _CTRL_W + _CTRL2_W
    on_edge_panel = (_EP_X <= x < _EP_X + ep_w)

    # ---- RIGHT-CLICK -------------------------------------------------------
    if event == cv2.EVENT_RBUTTONDOWN:
        if state.main_panel_h > 0 and y >= state.main_panel_h:
            return
        if on_edge_panel:
            dh, dw = state.warped_display.shape[:2]
            disp_x = max(0, min(x - _EP_X - _BORDER, dw - 1))
            disp_y = max(0, min(y - _BORDER, dh - 1))
            if state.local_mode:
                _exit_local_mode(state)
                state.edges_dirty = True
            elif state.merge_mode:
                _do_merge_rclick(state, disp_x, disp_y)
            else:
                for patch in reversed(state.patches):
                    if patch["mask"][disp_y, disp_x] > 0:
                        _unpatch_by_id(state, patch["patch_id"])
                        return
        return

    # ---- LEFT-CLICK --------------------------------------------------------
    if event == cv2.EVENT_LBUTTONDOWN:

        # Filmstrip (below main panels)
        if state.main_panel_h > 0 and y >= state.main_panel_h:
            if not state.local_mode and not state.merge_mode:
                _handle_filmstrip_click(state, x, y - state.main_panel_h)
            return

        if x < _CTRL_W:
            # ---- Control panel ---------------------------------------------
            if state.merge_mode:
                if state.merge_active_sa_id is not None:
                    for i, sl in enumerate(state.merge_sliders):
                        ty = _track_y(i)
                        if (x - sl.handle_x()) ** 2 + (y - ty) ** 2 <= (_HANDLE_R + 5) ** 2:
                            state.merge_drag_idx = i
                            return
                if _hit_button(x, y, _BTN_RESET_CX, _BTN_CY):
                    if state.merge_active_sa_id is not None:
                        for sl, iv in zip(state.merge_sliders, state.merge_init_vals):
                            sl.value = iv
                        _commit_sa_edit(state)
                        state.merge_dirty = True
                elif _hit_button(x, y, _BTN_TAKE_CX, _BTN_CY):
                    _do_take(state)
                elif _hit_button(x, y, _BTN_DONE_CX, _BTN_CY):
                    state.done = True
                elif _hit_button(x, y, _BTN_CLR_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    state.color_idx   = (state.color_idx + 1) % len(_EDGE_COLORS)
                    state.merge_dirty = True
                elif _hit_button(x, y, _CTRL_W // 2, _MERGE_BTN_CY, 160, _BTN_H):
                    if state.merge_active_sa_id is not None:
                        _exit_sa_edit(state)
                    state.merge_mode       = False
                    state.merge_dirty      = False
                    state.merge_edge_panel = None
                    state.edges_dirty      = True

            elif state.local_mode:
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
                    _commit_local_patch(state)
                    _exit_local_mode(state)
                    state.edges_dirty = True
                elif _hit_button(x, y, _BTN_CLR_CX, _OC_BTN_CY, _OC_BTN_W, _OC_BTN_H):
                    state.color_idx   = (state.color_idx + 1) % len(_EDGE_COLORS)
                    state.local_dirty = True
                elif _hit_button(x, y, _SEAL_MINUS_CX, _SEAL_ROW_Y,
                                 _SEAL_PBTN_W, _SEAL_PBTN_H):
                    if state.local_patch_idx is None:
                        state.local_seal = max(_SEAL_MIN, state.local_seal - 1)
                        _rerun_flood_fill(state)
                elif _hit_button(x, y, _SEAL_PLUS_CX, _SEAL_ROW_Y,
                                 _SEAL_PBTN_W, _SEAL_PBTN_H):
                    if state.local_patch_idx is None:
                        state.local_seal = min(_SEAL_MAX, state.local_seal + 1)
                        _rerun_flood_fill(state)
                elif _hit_button(x, y, _CTRL_W // 2, _SEAL_ROW_Y, 160, _SEAL_PBTN_H):
                    if state.local_patch_idx is not None:
                        del state.patches[state.local_patch_idx]
                        _exit_local_mode(state)
                        state.edges_dirty = True

            else:
                # Global mode
                for i, sl in enumerate(state.sliders):
                    ty = _track_y(i)
                    if (x - sl.handle_x()) ** 2 + (y - ty) ** 2 <= (_HANDLE_R + 5) ** 2:
                        state.drag_idx = i
                        return
                if _hit_button(x, y, _BTN_RESET_CX, _BTN_CY):
                    for sl, iv in zip(state.sliders, state.initial_values):
                        sl.value = iv
                    state.patches.clear()
                    state.super_areas.clear()
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
                elif (state.patches and
                      _hit_button(x, y, _CTRL_W // 2, _SEAL_ROW_Y, 100, _SEAL_PBTN_H)):
                    state.patches.clear()
                    state.super_areas.clear()
                    state.edges_dirty = True
                elif (_hit_button(x, y, _CTRL_W // 2, _MERGE_BTN_CY, 100, _BTN_H)
                      and len(state.patches) >= 1):
                    state.merge_mode  = True
                    state.merge_dirty = True
                    state.edges_dirty = True
                elif _hit_button(x, y, _BTN_PRINT_CX, _BTN_PRINT_CY,
                                 _BTN_PRINT_W, _BTN_PRINT_H):
                    if state.preview_take_idx is not None:
                        state.print_preview_take_idx = state.preview_take_idx
                        state.done = True

        elif on_edge_panel:
            # ---- Edge panel click ------------------------------------------
            dh, dw = state.warped_display.shape[:2]
            disp_x = max(0, min(x - _EP_X - _BORDER, dw - 1))
            disp_y = max(0, min(y - _BORDER, dh - 1))

            if state.merge_mode:
                _do_merge_lclick(state, disp_x, disp_y)
            elif state.local_mode:
                if state.local_mask is not None and state.local_mask[disp_y, disp_x] == 0:
                    _commit_local_patch(state)
                    _exit_local_mode(state)
                    state.edges_dirty = True
                else:
                    state.local_dirty = True
            else:
                if state.preview_take_idx is None:
                    _enter_local_mode(state, disp_x, disp_y)
                else:
                    if state.pre_seed_values is not None:
                        for sl, v in zip(state.sliders, state.pre_seed_values):
                            sl.value = v
                        state.pre_seed_values = None
                    state.preview_take_idx     = None
                    state.preview_color_ver_id = None
                    state.edges_dirty          = True

    # ---- MOUSE MOVE --------------------------------------------------------
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.merge_mode and state.merge_drag_idx is not None:
            sl = state.merge_sliders[state.merge_drag_idx]
            nv = sl.value_from_x(x)
            if nv != sl.value:
                sl.value = nv
                _commit_sa_edit(state)
                state.merge_dirty = True
        elif state.local_mode and state.local_drag_idx is not None:
            sl = state.local_sliders[state.local_drag_idx]
            nv = sl.value_from_x(x)
            if nv != sl.value:
                sl.value          = nv
                state.local_dirty = True
        elif not state.local_mode and not state.merge_mode and state.drag_idx is not None:
            sl = state.sliders[state.drag_idx]
            nv = sl.value_from_x(x)
            if nv != sl.value:
                sl.value          = nv
                state.edges_dirty = True

    # ---- BUTTON UP ---------------------------------------------------------
    elif event == cv2.EVENT_LBUTTONUP:
        state.drag_idx       = None
        state.local_drag_idx = None
        state.merge_drag_idx = None


# ---------------------------------------------------------------------------
# Take helper
# ---------------------------------------------------------------------------

def _do_take(state: _EdgemapState) -> None:
    """Compute a full-resolution edge map and append a _TakeEntry to state.takes."""
    h_full, w_full = state.warped_full.shape[:2]
    global_vals    = tuple(sl.value for sl in state.sliders)

    composite_base = _compute_composite_inv_gray(
        state.warped_display, state.sliders, state.patches, state.super_areas)

    if state.local_mode and state.local_mask is not None and state.local_sliders:
        l_vals    = [sl.value for sl in state.local_sliders]
        local_inv = compute_lab_edges(state.warped_display, *l_vals)
        merged    = composite_base.copy()
        merged[state.local_mask > 0] = local_inv[state.local_mask > 0]
        display_inv_gray = merged

        rx = w_full / max(1, state.warped_display.shape[1])
        ry = h_full / max(1, state.warped_display.shape[0])
        full_inv_global = compute_lab_edges(state.warped_full, *global_vals)
        full_inv_local  = compute_lab_edges(state.warped_full, *l_vals)
        full_edges      = full_inv_global.copy()

        sdx, sdy = state.local_seed_disp if state.local_seed_disp else (0, 0)
        local_info: dict | None = None
        if state.local_bbox is not None:
            x0, y0, x1, y1 = state.local_bbox
            # Scale mask to full resolution
            mask_full = cv2.resize(state.local_mask, (w_full, h_full),
                                   interpolation=cv2.INTER_NEAREST)
            full_edges[mask_full > 0] = full_inv_local[mask_full > 0]
            local_info = {
                "seed": (int(sdx * rx), int(sdy * ry)),
                "bbox": (int(x0 * rx), int(y0 * ry),
                         int(x1 * rx), int(y1 * ry)),
                "thresholds": l_vals,
            }
    else:
        display_inv_gray = composite_base
        full_edges       = cv2.resize(composite_base, (w_full, h_full),
                                      interpolation=cv2.INTER_NEAREST)
        local_info = None

    next_idx = max((t.index for t in state.takes), default=-1) + 1
    thumb    = _generate_thumbnail(display_inv_gray)

    entry = _TakeEntry(
        index             = next_idx,
        edges_full        = full_edges,
        display_inv_gray  = display_inv_gray,
        global_thresholds = global_vals,
        local_info        = local_info,
        seeded_from       = state.seeded_from,
        base_image        = state.base_image,
        thumbnail         = thumb,
        patches_snapshot  = _patches_to_session_data(state.patches),
        is_new            = True,
    )
    state.takes.append(entry)
    state.has_take_zero    = True
    state.preview_take_idx = None


# ===========================================================================
# Colorize editor  (Phase 3)
# ===========================================================================

@dataclass
class _ColorizeState:
    warped_full:     np.ndarray
    warped_display:  np.ndarray
    take_inv_gray:   np.ndarray
    inv_gray_full:   np.ndarray
    lab_patches:     list
    super_areas:     list
    sliders:         list
    painted_display: np.ndarray = field(default_factory=lambda: np.zeros((1,1,3), np.uint8))
    pending_mask:    Optional[np.ndarray] = None
    pending_seed:    Optional[tuple]      = None
    dominant_hsv:    Optional[tuple]      = None
    local_seal:      int  = _SEAL_DEFAULT
    committed:       list = field(default_factory=list)
    done:            bool = False
    saved:           bool = False
    left_dirty:      bool = True
    right_dirty:     bool = True
    left_panel:      Optional[np.ndarray] = None
    right_panel:     Optional[np.ndarray] = None
    drag_idx:        Optional[int] = None


def _resolve_clr_mask(cs: _ColorizeState, dx: int, dy: int) -> Optional[np.ndarray]:
    """SA > standalone patch > flood fill against Take composite edge map."""
    for sa in cs.super_areas:
        combined = None
        for pid in sa["patch_ids"]:
            p = next((p for p in cs.lab_patches if p.get("patch_id") == pid), None)
            if p is not None:
                combined = p["mask"] if combined is None else np.maximum(combined, p["mask"])
        if combined is not None and combined[dy, dx] > 0:
            return combined.copy()
    for p in reversed(cs.lab_patches):
        if p.get("super_area_id") is None and p["mask"][dy, dx] > 0:
            return p["mask"].copy()
    return _flood_fill_region(cs.take_inv_gray, dx, dy, seal_px=cs.local_seal)


def _apply_hsv_abs(img: np.ndarray, mask: np.ndarray, h_t: int, s_t: int, v_pct: int) -> np.ndarray:
    """Set H and S absolutely, scale V, only where mask > 0."""
    out  = img.copy()
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return out
    hsvf = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsvf[ys, xs, 0] = float(h_t)
    hsvf[ys, xs, 1] = float(s_t)
    hsvf[ys, xs, 2] = np.clip(hsvf[ys, xs, 2] * (v_pct / 100.0), 0, 255)
    return cv2.cvtColor(np.clip(hsvf, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def _make_clr_left(cs: _ColorizeState) -> np.ndarray:
    img = np.full((*cs.take_inv_gray.shape, 3), (255, 255, 255), dtype=np.uint8)
    img[cs.take_inv_gray == 0] = (0, 0, 0)

    if cs.pending_mask is not None:
        colored = _apply_hsv_abs(cs.warped_display, cs.pending_mask,
                                  cs.sliders[0].value, cs.sliders[1].value, cs.sliders[2].value)
        img[cs.pending_mask > 0] = colored[cs.pending_mask > 0]
        kern    = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(cs.pending_mask, kern, iterations=1)
        img[(dilated > 0) & (cs.pending_mask == 0) & (cs.take_inv_gray > 0)] = (180, 180, 60)

    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (40, 40, 80), -1)
    cv2.putText(img,
                f"EDGE  \u2014  click to select region  |  committed: {len(cs.committed)}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return _add_border(img, _BORDER)


def _make_clr_right(cs: _ColorizeState) -> np.ndarray:
    img = cs.painted_display.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (40, 60, 40), -1)
    cv2.putText(img, "PAINTED  \u2014  A=apply  S=save  R=reset  Esc=cancel",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return _add_border(img, _BORDER, color=_ROSE)


def _draw_clr_ctrl(cs: _ColorizeState) -> np.ndarray:
    W = _CTRL_W
    panel = np.full((_CLR_TOTAL_H, W, 3), 28, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (W, _CLR_CTRL_TOP - 4), (80, 40, 120), -1)
    cv2.putText(panel, "HSV Target", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    for i, sl in enumerate(cs.sliders):
        ty = _CLR_CTRL_TOP + i * _CLR_BAND_H + 58
        if i > 0:
            cv2.line(panel, (10, _CLR_CTRL_TOP + i*_CLR_BAND_H - 6),
                     (W-10, _CLR_CTRL_TOP + i*_CLR_BAND_H - 6), (60, 60, 60), 1)
        cv2.putText(panel, sl.label, (_TRACK_X0, ty-26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, sl.color, 1, cv2.LINE_AA)
        cv2.line(panel, (_TRACK_X0, ty), (_TRACK_X1, ty), (90, 90, 90), 2)
        for tx2 in (_TRACK_X0, _TRACK_X1):
            cv2.line(panel, (tx2, ty-5), (tx2, ty+5), (70, 70, 70), 1)
        hx = sl.handle_x()
        cv2.circle(panel, (hx, ty), _HANDLE_R, sl.color, -1)
        cv2.circle(panel, (hx, ty), _HANDLE_R+1, (240, 240, 240), 1)
        vs = str(sl.value)
        (tw, _), _ = cv2.getTextSize(vs, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(panel, vs, (_TRACK_X1-tw, ty+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    if cs.dominant_hsv is not None:
        dh, ds, dv = cs.dominant_hsv
        dbgr = cv2.cvtColor(np.array([[[dh, ds, dv]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        sw_y = _CLR_BTN_CY - 22
        cv2.rectangle(panel, (W//2-44, sw_y-10), (W//2+44, sw_y+10),
                      (int(dbgr[0]), int(dbgr[1]), int(dbgr[2])), -1)
        cv2.rectangle(panel, (W//2-44, sw_y-10), (W//2+44, sw_y+10), (160, 160, 160), 1)
        info = f"H:{dh}  S:{ds}  V:{dv}"
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.putText(panel, info, (W//2-tw//2, sw_y+24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)

    _draw_button(panel, "RESET", _CLR_RESET_CX, _CLR_BTN_CY, (90, 90, 30))
    _draw_button(panel, "APPLY", _CLR_APPLY_CX, _CLR_BTN_CY, (35, 130, 35))
    _draw_button(panel, "-", _SEAL_MINUS_CX, _CLR_SEAL_ROW_Y, (60,60,80), _SEAL_PBTN_W, _SEAL_PBTN_H)
    _draw_button(panel, "+", _SEAL_PLUS_CX,  _CLR_SEAL_ROW_Y, (60,60,80), _SEAL_PBTN_W, _SEAL_PBTN_H)
    st = f"SEAL: {cs.local_seal}"
    (tw, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    cv2.putText(panel, st, (W//2-tw//2, _CLR_SEAL_ROW_Y+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, _AMBER, 1, cv2.LINE_AA)
    _draw_button(panel, "SAVE", W//2, _CLR_SAVE_CY,
                 (0, 140, 80) if cs.committed else (50, 50, 50), 120, _BTN_H)
    if cs.committed:
        _draw_button(panel, "CLR ALL", W//2, _CLR_CLR_CY, (60, 40, 40), 120, _SEAL_PBTN_H)
    elif cs.pending_mask is not None:
        ht = "Adjust HSV then APPLY"
        (tw, _), _ = cv2.getTextSize(ht, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(panel, ht, (W//2-tw//2, _CLR_CLR_CY+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _AMBER, 1, cv2.LINE_AA)
    else:
        ht2 = "Click edge map to select"
        (tw, _), _ = cv2.getTextSize(ht2, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(panel, ht2, (W//2-tw//2, _CLR_CLR_CY+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
    return panel


def _clr_apply(cs: _ColorizeState) -> None:
    if cs.pending_mask is None:
        return
    h, w = cs.warped_display.shape[:2]
    sx, sy = cs.pending_seed
    cs.painted_display = _apply_hsv_abs(
        cs.painted_display, cs.pending_mask,
        cs.sliders[0].value, cs.sliders[1].value, cs.sliders[2].value)
    cs.committed.append({
        "seed_norm": (sx / max(1,w), sy / max(1,h)),
        "seal": cs.local_seal,
        "hsv": (cs.sliders[0].value, cs.sliders[1].value, cs.sliders[2].value),
    })
    cs.pending_mask = None; cs.pending_seed = None; cs.dominant_hsv = None
    for sl, d in zip(cs.sliders, _CLR_SLIDER_DEFS): sl.value = d[3]
    cs.left_dirty = cs.right_dirty = True


def _clr_rerun(cs: _ColorizeState) -> None:
    if cs.pending_seed is None: return
    sx, sy = cs.pending_seed
    m = _resolve_clr_mask(cs, sx, sy)
    if m is not None:
        cs.pending_mask = m; cs.left_dirty = cs.right_dirty = True


def _colorize_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    cs: _ColorizeState = param
    lw    = cs.take_inv_gray.shape[1] + 2 * _BORDER
    on_L  = (_CTRL_W <= x < _CTRL_W + lw)
    on_R  = (_CTRL_W + lw <= x)

    if event == cv2.EVENT_RBUTTONDOWN and on_R and cs.committed:
        cs.committed.pop()
        cs.painted_display = cs.warped_display.copy()
        for rec in cs.committed:
            nx, ny = rec["seed_norm"]
            dh2, dw2 = cs.take_inv_gray.shape[:2]
            sx2 = max(0, min(int(round(nx*dw2)), dw2-1))
            sy2 = max(0, min(int(round(ny*dh2)), dh2-1))
            m2 = _resolve_clr_mask(cs, sx2, sy2)
            if m2 is not None:
                cs.painted_display = _apply_hsv_abs(cs.painted_display, m2, *rec["hsv"])
        cs.right_dirty = True

    elif event == cv2.EVENT_LBUTTONDOWN:
        if x < _CTRL_W:
            for i, sl in enumerate(cs.sliders):
                ty = _CLR_CTRL_TOP + i * _CLR_BAND_H + 58
                if (x-sl.handle_x())**2 + (y-ty)**2 <= (_HANDLE_R+5)**2:
                    cs.drag_idx = i; return
            if _hit_button(x, y, _CLR_RESET_CX, _CLR_BTN_CY):
                for sl, d in zip(cs.sliders, _CLR_SLIDER_DEFS): sl.value = d[3]
                cs.left_dirty = cs.right_dirty = True
            elif _hit_button(x, y, _CLR_APPLY_CX, _CLR_BTN_CY): _clr_apply(cs)
            elif _hit_button(x, y, _SEAL_MINUS_CX, _CLR_SEAL_ROW_Y, _SEAL_PBTN_W, _SEAL_PBTN_H):
                cs.local_seal = max(_SEAL_MIN, cs.local_seal-1); _clr_rerun(cs)
            elif _hit_button(x, y, _SEAL_PLUS_CX, _CLR_SEAL_ROW_Y, _SEAL_PBTN_W, _SEAL_PBTN_H):
                cs.local_seal = min(_SEAL_MAX, cs.local_seal+1); _clr_rerun(cs)
            elif _hit_button(x, y, _CTRL_W//2, _CLR_SAVE_CY, 120, _BTN_H):
                if cs.committed: cs.saved = True; cs.done = True
            elif cs.committed and _hit_button(x, y, _CTRL_W//2, _CLR_CLR_CY, 120, _SEAL_PBTN_H):
                cs.committed.clear()
                cs.painted_display = cs.warped_display.copy()
                cs.pending_mask = None; cs.pending_seed = None; cs.dominant_hsv = None
                cs.left_dirty = cs.right_dirty = True
        elif on_L:
            dh2, dw2 = cs.take_inv_gray.shape[:2]
            dx = max(0, min(x - _CTRL_W - _BORDER, dw2-1))
            dy = max(0, min(y - _BORDER, dh2-1))
            mask = _resolve_clr_mask(cs, dx, dy)
            if mask is not None:
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    hsv_img = cv2.cvtColor(cs.warped_display, cv2.COLOR_BGR2HSV)
                    h_vals  = hsv_img[ys, xs, 0].astype(np.float32)
                    s_vals  = hsv_img[ys, xs, 1].astype(np.float32)
                    v_vals  = hsv_img[ys, xs, 2].astype(np.float32)
                    h_sin   = float(np.mean(np.sin(h_vals * np.pi / 90.0)))
                    h_cos   = float(np.mean(np.cos(h_vals * np.pi / 90.0)))
                    dom_h   = int(np.degrees(np.arctan2(h_sin, h_cos)) * 90.0 / 180.0) % 180
                    dom_s   = int(np.median(s_vals))
                    dom_v   = int(np.median(v_vals))
                    cs.sliders[0].value = dom_h
                    cs.sliders[1].value = dom_s
                    cs.sliders[2].value = 100
                    cs.dominant_hsv = (dom_h, dom_s, dom_v)
                cs.pending_mask = mask; cs.pending_seed = (dx, dy)
                cs.left_dirty = cs.right_dirty = True

    elif event == cv2.EVENT_MOUSEMOVE and cs.drag_idx is not None:
        sl = cs.sliders[cs.drag_idx]
        nv = sl.value_from_x(x)
        if nv != sl.value:
            sl.value = nv; cs.left_dirty = cs.right_dirty = True

    elif event == cv2.EVENT_LBUTTONUP:
        cs.drag_idx = None


def _committed_to_session_data(committed: list) -> list:
    return [{"seed_norm": list(r["seed_norm"]), "seal": r["seal"], "hsv": list(r["hsv"])}
            for r in committed]


def _apply_color_full_res(
    base_full: np.ndarray,
    inv_gray_full: np.ndarray,
    committed: list,
    full_lab_patches: list,
    super_areas: list,
) -> np.ndarray:
    out = base_full.copy()
    for rec in committed:
        try:
            nx, ny = rec["seed_norm"]; seal = int(rec.get("seal", _SEAL_DEFAULT))
            h_t, s_t, v_pct = rec["hsv"]
            h_f, w_f = out.shape[:2]
            sx = max(0, min(int(round(nx*w_f)), w_f-1))
            sy = max(0, min(int(round(ny*h_f)), h_f-1))
            mask = None
            for sa in super_areas:
                comb = None
                for pid in sa["patch_ids"]:
                    p = next((p for p in full_lab_patches if p.get("patch_id")==pid), None)
                    if p is not None:
                        comb = p["mask"] if comb is None else np.maximum(comb, p["mask"])
                if comb is not None and comb[sy, sx] > 0:
                    mask = comb; break
            if mask is None:
                for p in reversed(full_lab_patches):
                    if p.get("super_area_id") is None and p["mask"][sy,sx] > 0:
                        mask = p["mask"]; break
            if mask is None:
                mask = _flood_fill_region(inv_gray_full, sx, sy, seal_px=seal)
            out = _apply_hsv_abs(out, mask, h_t, s_t, v_pct)
        except Exception as exc:
            print(f"[WARN] colorize full-res: {exc!r}")
    return out


def edit_colorize(
    warped_full: np.ndarray,
    take_inv_gray_disp: np.ndarray,
    take_inv_gray_full: np.ndarray,
    lab_patches: list,
    super_areas: list,
    base_display: np.ndarray | None = None,
    base_full: np.ndarray | None = None,
    initial_committed: list | None = None,
) -> tuple[np.ndarray | None, list]:
    """Two-panel colorize editor."""
    sliders = [
        _Slider(label=d[0], min_val=d[1], max_val=d[2], value=d[3], color=d[4])
        for d in _CLR_SLIDER_DEFS
    ]
    disp_w, disp_h = _compute_panel_size_colorize(warped_full)
    interp         = cv2.INTER_AREA if disp_w < warped_full.shape[1] else cv2.INTER_LINEAR
    warped_disp    = cv2.resize(warped_full, (disp_w, disp_h), interpolation=interp)
    painted_start  = (cv2.resize(base_display, (disp_w, disp_h), interpolation=interp)
                      if base_display is not None else warped_disp.copy())

    inv_d = cv2.resize(take_inv_gray_disp, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
    disp_patches: list = []
    for p in (lab_patches or []):
        sp = dict(p)
        sp["mask"] = cv2.resize(p["mask"], (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        disp_patches.append(sp)

    cs = _ColorizeState(
        warped_full    = warped_full,
        warped_display = warped_disp,
        take_inv_gray  = inv_d,
        inv_gray_full  = take_inv_gray_full,
        lab_patches    = disp_patches,
        super_areas    = super_areas or [],
        sliders        = sliders,
        painted_display = painted_start,
    )

    if initial_committed:
        for rec in initial_committed:
            try:
                nx, ny = rec["seed_norm"]; seal = int(rec.get("seal", _SEAL_DEFAULT))
                sx = max(0, min(int(round(nx*disp_w)), disp_w-1))
                sy = max(0, min(int(round(ny*disp_h)), disp_h-1))
                m  = _resolve_clr_mask(cs, sx, sy)
                if m is None: continue
                h_t, s_t, v_pct = rec["hsv"]
                cs.painted_display = _apply_hsv_abs(cs.painted_display, m, h_t, s_t, v_pct)
                cs.committed.append(rec)
            except Exception:
                continue

    cv2.namedWindow(_CLR_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_CLR_WINDOW, _colorize_mouse, cs)

    try:
        while not cs.done:
            if cs.left_dirty:
                cs.left_panel  = _make_clr_left(cs)
                cs.left_dirty  = False
            if cs.right_dirty:
                cs.right_panel = _make_clr_right(cs)
                cs.right_dirty = False
            ctrl  = _draw_clr_ctrl(cs)
            h_max = max(ctrl.shape[0], cs.left_panel.shape[0], cs.right_panel.shape[0])
            frame = np.hstack([
                _pad_to_height(ctrl,            h_max),
                _pad_to_height(cs.left_panel,   h_max),
                _pad_to_height(cs.right_panel,  h_max),
            ])
            cv2.imshow(_CLR_WINDOW, frame)
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(_CLR_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("a"), ord("A")): 
                _clr_apply(cs)
            elif key in (ord("r"), ord("R")):
                for sl, d in zip(cs.sliders, _CLR_SLIDER_DEFS): sl.value = d[3]
                cs.left_dirty = cs.right_dirty = True
            elif key in (ord("s"), ord("S")):
                if cs.committed: cs.saved = True; cs.done = True
            elif key in (27, ord("q"), ord("Q")): cs.done = True
    finally:
        try:
            cv2.destroyWindow(_CLR_WINDOW)
        except Exception:
            pass

    if not cs.saved or not cs.committed:
        return None, []

    print("[INFO] Applying color at full resolution...")
    h_f, w_f = warped_full.shape[:2]
    base_fr  = base_full if base_full is not None else warped_full
    full_pts: list = []
    for p in (lab_patches or []):
        fp = dict(p)
        fp["mask"] = cv2.resize(p["mask"], (w_f, h_f), interpolation=cv2.INTER_NEAREST)
        full_pts.append(fp)
    result = _apply_color_full_res(base_fr, cs.inv_gray_full, cs.committed, full_pts, super_areas or [])
    return result, _committed_to_session_data(cs.committed)


# ===========================================================================
# Print Preview modal
# ===========================================================================

_PP_WINDOW     = "paintscan - print preview  (S=save  Esc=close)"
_PP_CTRL_W     = 300
_PP_DISP_H     = 700   # target display height for centre and right panels

_PP_GRAY_SLIDER_Y = 80
_PP_BORDER_BTN_CY = 180
_PP_GRAY_BTN_CY   = 218
_PP_OVL_BTN_CY    = 256
_PP_SAVE_BTN_CY   = 330
_PP_CLOSE_BTN_CY  = 372
_PP_BTN_W, _PP_BTN_H = 180, 30

_PP_SLIDER_X0      = 20
_PP_SLIDER_X1      = _PP_CTRL_W - 20
_PP_THICK_SLIDER_Y = 118   # line thickness slider Y position


@dataclass
class _PPState:
    display_inv_gray: np.ndarray   # editor-res inv_gray — for center panel display (no thickening)
    edges_full:       np.ndarray   # full-res edges — for save-time thickening only
    gray_disp:        np.ndarray   # editor-res BGR gray painting — for right panel display
    gray_full:        np.ndarray   # full-res BGR gray painting — for save
    take_idx:         int
    brightness:       int  = 100   # 50–200, alpha = brightness/100
    thickness:        int  = 2     # 0–8, dilation px at display res; scaled proportionally for save
    overlay_on:       bool = False
    save_target:      str  = "border"   # "border" | "gray"
    done:             bool = False
    dirty:            bool = True
    ctrl_panel:       Optional[np.ndarray] = None
    center_panel:     Optional[np.ndarray] = None
    right_panel:      Optional[np.ndarray] = None


def _make_border_print(inv_gray: np.ndarray, dil_px: int) -> np.ndarray:
    """Apply dil_px-pixel dilation to inv_gray; return BGR black-on-white.
    Called for center panel display (on display_inv_gray) and at save time
    (on edges_full with dil_px scaled to full resolution)."""
    out = np.full((*inv_gray.shape, 3), 255, dtype=np.uint8)
    if dil_px > 0:
        kernel     = np.ones((dil_px * 2 + 1, dil_px * 2 + 1), np.uint8)
        edge_mask  = cv2.bitwise_not(inv_gray)
        thick_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        out[thick_mask > 0] = 0
    else:
        out[inv_gray == 0] = 0
    return out


def _pp_render(ps: _PPState, disp_w: int, disp_h: int) -> None:
    """Rebuild all three display panels into ps.*_panel."""
    ih, iw = ps.display_inv_gray.shape[:2]
    scale  = min(disp_w / max(iw, 1), disp_h / max(ih, 1))
    cw     = max(1, int(round(iw * scale)))
    ch     = max(1, int(round(ih * scale)))

    # --- Centre: thickened edge map — WYSIWYG of what will be saved and printed ---
    inv_scaled = cv2.resize(ps.display_inv_gray, (cw, ch), interpolation=cv2.INTER_AREA)
    center_bgr = _make_border_print(inv_scaled, ps.thickness)
    cp  = np.full((disp_h, disp_w, 3), 200, dtype=np.uint8)
    y0c = (disp_h - ch) // 2; x0c = (disp_w - cw) // 2
    cp[y0c:y0c+ch, x0c:x0c+cw] = center_bgr
    ps.center_panel = cp

    # --- Right: gray painting + optional overlay (same thickening as centre) ---
    alpha  = ps.brightness / 100.0
    gh, gw = ps.gray_disp.shape[:2]
    gscale = min(disp_w / max(gw, 1), disp_h / max(gh, 1))
    rw     = max(1, int(round(gw * gscale)))
    rh     = max(1, int(round(gh * gscale)))
    gray_b = cv2.convertScaleAbs(
        cv2.resize(ps.gray_disp, (rw, rh), interpolation=cv2.INTER_AREA),
        alpha=alpha, beta=0)
    if ps.overlay_on:
        inv_ovl    = cv2.resize(ps.display_inv_gray, (rw, rh), interpolation=cv2.INTER_NEAREST)
        border_ovl = _make_border_print(inv_ovl, ps.thickness)
        gray_b[np.all(border_ovl == 0, axis=2)] = 0
    rp  = np.full((disp_h, disp_w, 3), 40, dtype=np.uint8)
    ry0 = (disp_h - rh) // 2; rx0 = (disp_w - rw) // 2
    rp[ry0:ry0+rh, rx0:rx0+rw] = gray_b
    ps.right_panel = rp

    # --- Left: controls ---
    ctrl = np.full((disp_h, _PP_CTRL_W, 3), 28, dtype=np.uint8)

    cv2.rectangle(ctrl, (0, 0), (_PP_CTRL_W, 28), (50, 40, 80), -1)
    cv2.putText(ctrl, f"PRINT PREVIEW  T{ps.take_idx}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(ctrl, "Gray brightness", (10, _PP_GRAY_SLIDER_Y - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,160,160), 1, cv2.LINE_AA)
    cv2.line(ctrl, (_PP_SLIDER_X0, _PP_GRAY_SLIDER_Y),
             (_PP_SLIDER_X1, _PP_GRAY_SLIDER_Y), (80,80,80), 2)
    hx = _PP_SLIDER_X0 + int((ps.brightness - 50) / 150 * (_PP_SLIDER_X1 - _PP_SLIDER_X0))
    cv2.circle(ctrl, (hx, _PP_GRAY_SLIDER_Y), 9, (180,180,180), -1)
    cv2.circle(ctrl, (hx, _PP_GRAY_SLIDER_Y), 10, (220,220,220), 1)
    cv2.putText(ctrl, f"{ps.brightness}%", (_PP_SLIDER_X0, _PP_GRAY_SLIDER_Y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120,120,120), 1, cv2.LINE_AA)

    cv2.putText(ctrl, "Line thickness", (10, _PP_THICK_SLIDER_Y - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160,160,160), 1, cv2.LINE_AA)
    cv2.line(ctrl, (_PP_SLIDER_X0, _PP_THICK_SLIDER_Y),
             (_PP_SLIDER_X1, _PP_THICK_SLIDER_Y), (80,80,80), 2)
    thx = _PP_SLIDER_X0 + int(ps.thickness / 8 * (_PP_SLIDER_X1 - _PP_SLIDER_X0))
    cv2.circle(ctrl, (thx, _PP_THICK_SLIDER_Y), 9, (160,160,200), -1)
    cv2.circle(ctrl, (thx, _PP_THICK_SLIDER_Y), 10, (220,220,220), 1)
    cv2.putText(ctrl, f"{ps.thickness}px", (_PP_SLIDER_X0, _PP_THICK_SLIDER_Y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120,120,120), 1, cv2.LINE_AA)

    cv2.line(ctrl, (10, 148), (_PP_CTRL_W-10, 148), (55,55,55), 1)
    cv2.putText(ctrl, "Save target:", (10, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120,120,120), 1, cv2.LINE_AA)

    b_col = (35,130,35) if ps.save_target == "border" else (55,55,55)
    g_col = (35,130,35) if ps.save_target == "gray"   else (55,55,55)
    o_col = (0,140,200) if ps.overlay_on               else (55,55,55)
    _draw_button(ctrl, "BORDER",  _PP_CTRL_W//2, _PP_BORDER_BTN_CY, b_col, _PP_BTN_W, _PP_BTN_H)
    _draw_button(ctrl, "GRAY",    _PP_CTRL_W//2, _PP_GRAY_BTN_CY,   g_col, _PP_BTN_W, _PP_BTN_H)
    _draw_button(ctrl, "OVERLAY", _PP_CTRL_W//2, _PP_OVL_BTN_CY,   o_col, _PP_BTN_W, _PP_BTN_H)

    if ps.overlay_on and ps.save_target == "gray":
        hint = "Overlay applied on save"
        (hw,_),_ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(ctrl, hint, (_PP_CTRL_W//2-hw//2, _PP_OVL_BTN_CY+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0,170,230), 1, cv2.LINE_AA)

    cv2.line(ctrl, (10, _PP_SAVE_BTN_CY-20), (_PP_CTRL_W-10, _PP_SAVE_BTN_CY-20), (55,55,55), 1)
    _draw_button(ctrl, "SAVE",  _PP_CTRL_W//2, _PP_SAVE_BTN_CY,  (0, 100, 180), _PP_BTN_W, _PP_BTN_H)
    _draw_button(ctrl, "CLOSE", _PP_CTRL_W//2, _PP_CLOSE_BTN_CY, (70,  70,  70), _PP_BTN_W, _PP_BTN_H)

    ps.ctrl_panel = ctrl
    ps.dirty      = False


def _pp_save(ps: _PPState, out_dir: Path, stem: str, jpg_quality: int) -> None:
    """Apply thickening at save time only."""
    alpha  = ps.brightness / 100.0
    gray_b = cv2.convertScaleAbs(ps.gray_full, alpha=alpha, beta=0)

    # Scale thickness from display resolution to full resolution for save
    disp_long = max(ps.display_inv_gray.shape[:2])
    full_long  = max(ps.edges_full.shape[:2])
    save_dil   = max(1, round(ps.thickness * full_long / disp_long)) if ps.thickness > 0 else 0

    if ps.save_target == "border":
        border_print = _make_border_print(ps.edges_full, save_dil)
        path = out_dir / f"{stem}_print_border_T{ps.take_idx}.png"
        cv2.imwrite(str(path), border_print)
        print(f"[INFO] Print border saved: {path.name}")

    elif ps.save_target == "gray":
        if ps.overlay_on:
            border_print = _make_border_print(ps.edges_full, save_dil)
            gh, gw  = ps.gray_full.shape[:2]
            brd_r   = cv2.resize(border_print, (gw, gh), interpolation=cv2.INTER_NEAREST)
            edge_px = np.all(brd_r == 0, axis=2)
            gray_b[edge_px] = 0
            path = out_dir / f"{stem}_print_overlay_T{ps.take_idx}.png"
            cv2.imwrite(str(path), gray_b)
            print(f"[INFO] Print overlay saved: {path.name}")
        else:
            path = out_dir / f"{stem}_print_gray_T{ps.take_idx}.jpg"
            cv2.imwrite(str(path), gray_b, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            print(f"[INFO] Print gray saved: {path.name}")


def _run_print_preview(
    state: _EdgemapState,
    out_dir: Path,
    stem: str,
    jpg_quality: int,
) -> None:
    """Open the Print Preview modal window.  Blocking; returns when user closes."""
    take_entry = next(
        (t for t in state.takes if t.index == state.print_preview_take_idx), None
    )
    if take_entry is None:
        print("[WARN] Print preview: Take not found.")
        return

    # T0 loaded from a prior session has edges_full=None — recompute from thresholds.
    edges_full = take_entry.edges_full
    if edges_full is None:
        print(f"[INFO] Print preview: recomputing full-res edges for T{take_entry.index}...")
        edges_full = compute_lab_edges(state.warped_full, *take_entry.global_thresholds)

    # Painting source: selected color version from disk, or master
    painting_full = state.warped_full.copy()
    if state.preview_color_ver_id is not None:
        cv_path = out_dir / f"{stem}_color_v{state.preview_color_ver_id}.jpg"
        if cv_path.exists():
            loaded = cv2.imread(str(cv_path))
            if loaded is not None:
                painting_full = loaded

    gray_full = cv2.cvtColor(
        cv2.cvtColor(painting_full, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # Display-res gray painting scaled to match display_inv_gray dimensions
    dh, dw    = take_entry.display_inv_gray.shape[:2]
    paint_d   = cv2.resize(painting_full, (dw, dh), interpolation=cv2.INTER_AREA)
    gray_disp = cv2.cvtColor(
        cv2.cvtColor(paint_d, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    ps = _PPState(
        display_inv_gray = take_entry.display_inv_gray,
        edges_full       = edges_full,
        gray_disp        = gray_disp,
        gray_full        = gray_full,
        take_idx         = take_entry.index,
    )

    ih, iw = take_entry.display_inv_gray.shape[:2]
    disp_h  = _PP_DISP_H
    disp_w  = max(200, int(round(iw * disp_h / max(ih, 1))))

    drag = [False]   # mutable container so nested _pp_mouse can mutate it

    def _hit_pp(mx, my, cx, cy, w=_PP_BTN_W, h=_PP_BTN_H) -> bool:
        return abs(mx - cx) <= w // 2 and abs(my - cy) <= h // 2

    def _pp_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if mx < _PP_CTRL_W:
                if abs(my - _PP_GRAY_SLIDER_Y) <= 14 and _PP_SLIDER_X0 <= mx <= _PP_SLIDER_X1:
                    drag[0] = (1, 'brightness')
                    ratio = (mx - _PP_SLIDER_X0) / max(1, _PP_SLIDER_X1 - _PP_SLIDER_X0)
                    ps.brightness = int(round(50 + ratio * 150))
                    ps.dirty = True
                elif abs(my - _PP_THICK_SLIDER_Y) <= 14 and _PP_SLIDER_X0 <= mx <= _PP_SLIDER_X1:
                    drag[0] = (1, 'thickness')
                    ratio = (mx - _PP_SLIDER_X0) / max(1, _PP_SLIDER_X1 - _PP_SLIDER_X0)
                    ps.thickness = int(round(ratio * 8))
                    ps.dirty = True
                elif _hit_pp(mx, my, _PP_CTRL_W//2, _PP_BORDER_BTN_CY):
                    ps.save_target = "border"; ps.dirty = True
                elif _hit_pp(mx, my, _PP_CTRL_W//2, _PP_GRAY_BTN_CY):
                    ps.save_target = "gray";   ps.dirty = True
                elif _hit_pp(mx, my, _PP_CTRL_W//2, _PP_OVL_BTN_CY):
                    ps.overlay_on = not ps.overlay_on; ps.dirty = True
                elif _hit_pp(mx, my, _PP_CTRL_W//2, _PP_SAVE_BTN_CY):
                    _pp_save(ps, out_dir, stem, jpg_quality)
                elif _hit_pp(mx, my, _PP_CTRL_W//2, _PP_CLOSE_BTN_CY):
                    ps.done = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag[0] and _PP_SLIDER_X0 <= mx <= _PP_SLIDER_X1:
                ratio = (mx - _PP_SLIDER_X0) / max(1, _PP_SLIDER_X1 - _PP_SLIDER_X0)
                if drag[0][1] == 'brightness':
                    ps.brightness = int(round(50 + ratio * 150))
                else:
                    ps.thickness  = int(round(ratio * 8))
                ps.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            drag[0] = False

    cv2.namedWindow(_PP_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_PP_WINDOW, _pp_mouse)

    try:
        while not ps.done:
            if ps.dirty:
                _pp_render(ps, disp_w, disp_h)
            frame = np.hstack([ps.ctrl_panel, ps.center_panel, ps.right_panel])
            cv2.imshow(_PP_WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(_PP_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27 or key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                _pp_save(ps, out_dir, stem, jpg_quality)
    finally:
        try:
            cv2.destroyWindow(_PP_WINDOW)
        except Exception:
            pass
        
# ===========================================================================
# Public entry point
# ===========================================================================

def edit_edgemap(
    warped: np.ndarray,
    l_lo: int = 50,
    l_hi: int = 150,
    a_lo: int = 30,
    a_hi: int = 90,
    b_lo: int = 30,
    b_hi: int = 90,
    base_image: str = "master",
    initial_patches_data: list | None = None,
    initial_takes_data: list | None = None,
    initial_super_areas_data: list | None = None,
    initial_color_versions: dict | None = None,
    initial_preview_take_idx: int | None = None,
    out_dir: Path | None = None,
    stem: str | None = None,
    jpg_quality: int = 92,
) -> tuple[list, list, list, Optional[int]]:
    """Open the three-panel Lab edge-map editor.

    Global mode
    -----------
    T / TAKE  = commit current thresholds as a new Take
    R / RESET = restore sliders to session-open values and clear patches
    O         = overlay toggle
    C         = cycle edge colour palette
    Esc/D/Q   = close session and return

    Local mode  (entered by clicking anywhere on the edge panel)
    ----------
    T / TAKE      = take with global-outside + local-inside merge
    R / RESET     = reset local sliders to their entry values
    L             = commit local patch and exit local mode
    Esc           = discard local work and exit local mode
    EXIT LOCAL    = same as L (commit + exit)

    Filmstrip  (strip below the three panels)
    ---------
    Click a thumbnail  = preview that Take's edge map in the middle panel
    Click again        = deselect (return to live)
    SEED button        = load previewed Take's thresholds into working sliders
    LIVE button        = return to live view without seeding

    Parameters
    ----------
    initial_patches_data:
        Serialised patches from a prior session, restored on re-entry.
    initial_takes_data:
        List of take dicts from a prior session.  When provided the filmstrip
        is pre-populated with historical takes and Take 0 is NOT auto-created
        (it already exists in the list).  When None a fresh Take 0 is recorded.
    out_dir, stem, jpg_quality:
        Required for Print Preview saves.  Safe to omit when not using PP.

    Returns
    -------
    (new_takes, patches_data, super_areas_data, colorize_take_idx)
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

    disp_w, disp_h = _compute_panel_size(warped)
    interp         = cv2.INTER_AREA if disp_w < warped.shape[1] else cv2.INTER_LINEAR
    master_scaled  = cv2.resize(warped, (disp_w, disp_h), interpolation=interp)
    master_panel   = _add_border(master_scaled, _BORDER)

    state = _EdgemapState(
        warped_full    = warped,
        warped_display = master_scaled,
        master_panel   = master_panel,
        sliders        = sliders,
        initial_values = list(initial_vals),
        base_image     = base_image,
        edges_dirty    = True,
    )

    # --- Populate filmstrip from prior session, or leave empty for fresh start ---
    if initial_takes_data:
        for td in initial_takes_data:
            thresholds = (
                int(td.get("l_lo", 50)),  int(td.get("l_hi", 150)),
                int(td.get("a_lo", 30)),  int(td.get("a_hi",  90)),
                int(td.get("b_lo", 30)),  int(td.get("b_hi",  90)),
            )
            inv_gray  = compute_lab_edges(master_scaled, *thresholds)
            thumb     = _generate_thumbnail(inv_gray)
            take_idx  = int(td.get("index", 0))
            cvs       = (initial_color_versions or {}).get(take_idx, [])
            state.takes.append(_TakeEntry(
                index             = take_idx,
                edges_full        = None,
                display_inv_gray  = inv_gray,
                global_thresholds = thresholds,
                local_info        = td.get("local_region"),
                seeded_from       = td.get("seeded_from"),
                base_image        = td.get("base_image", base_image),
                thumbnail         = thumb,
                patches_snapshot  = [],
                is_new            = False,
                color_versions    = cvs,
            ))
        state.has_take_zero = True
        print(f"[INFO] Filmstrip restored: {len(state.takes)} historical take(s)")
    else:
        state.has_take_zero = False

    # Pre-select a Take (e.g. returning from colorize)
    if initial_preview_take_idx is not None:
        entry0 = next((t for t in state.takes if t.index == initial_preview_take_idx), None)
        if entry0 is not None:
            state.preview_take_idx = initial_preview_take_idx
            state.pre_seed_values  = list(initial_vals)
            for sl, val in zip(state.sliders, entry0.global_thresholds):
                sl.value = val

    # --- Restore committed patches from a previous session ------------------
    if initial_patches_data:
        restored_patches, next_pid = _restore_patches_from_session(
            master_scaled, initial_patches_data, sliders)
        state.patches       = restored_patches
        state.next_patch_id = next_pid
        if state.patches:
            print(f"[INFO] Restored {len(state.patches)} local patch(es) from session")

    # --- Restore super-areas from a previous session ------------------------
    if initial_super_areas_data:
        state.super_areas = _restore_super_areas_from_session(initial_super_areas_data)
        if state.super_areas:
            max_said = max(s["super_area_id"] for s in state.super_areas)
            state.next_super_area_id = max_said + 1
            print(f"[INFO] Restored {len(state.super_areas)} super-area(s) from session")

    # Recompute historical take display_inv_gray with restored patches/SAs.
    if state.patches or state.super_areas:
        for take in state.takes:
            if not take.is_new:
                take_sliders = [
                    _Slider(d[0], d[1], d[2], max(d[1], min(d[2], v)), d[4])
                    for d, v in zip(_SLIDER_DEFS, take.global_thresholds)
                ]
                take.display_inv_gray = _compute_composite_inv_gray(
                    master_scaled, take_sliders, state.patches, state.super_areas)
                take.thumbnail = _generate_thumbnail(take.display_inv_gray)

    cv2.namedWindow(_EDGEMAP_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(_EDGEMAP_WINDOW, _edgemap_mouse, state)

    # Bootstrap — build the first frame, then lock the window to its exact size.
    _boot_ctrl = _draw_ctrl_panel(
        state.sliders,
        sum(1 for t in state.takes if t.index > 0),
        state.color_idx, state.overlay_on,
        has_take_zero         = state.has_take_zero,
        print_preview_enabled = (state.preview_take_idx is not None),
    )
    state.edge_panel, state.inv_gray_cache = _make_edge_panel(
        state.warped_display, state.sliders, state.color_idx,
        patches=state.patches, super_areas=state.super_areas)
    state.edges_dirty = False
    _bt_h = max(_boot_ctrl.shape[0], state.edge_panel.shape[0], master_panel.shape[0])
    state.main_panel_h = _bt_h
    _boot_info = _draw_info_panel(state, _bt_h)
    _boot_main = np.hstack([
        _pad_to_height(_boot_ctrl,       _bt_h),
        _pad_to_height(_boot_info,       _bt_h),
        _pad_to_height(state.edge_panel, _bt_h),
        _pad_to_height(master_panel,     _bt_h),
    ])
    state.window_w = _boot_main.shape[1]
    _boot_film  = _make_filmstrip_panel(state, state.window_w)
    boot_frame  = np.vstack([_boot_main, _boot_film])
    cv2.imshow(_EDGEMAP_WINDOW, boot_frame)
    cv2.resizeWindow(_EDGEMAP_WINDOW, boot_frame.shape[1], boot_frame.shape[0])
    cv2.waitKey(1)

    try:
        while True:
            # Rebuild panels when dirty
            if state.edges_dirty and not state.local_mode and not state.merge_mode:
                state.edge_panel, state.inv_gray_cache = _make_edge_panel(
                    state.warped_display, state.sliders, state.color_idx,
                    patches=state.patches, super_areas=state.super_areas)
                if state.overlay_on:
                    state.overlay_panel = _make_overlay_panel(
                        state.warped_display, state.sliders, state.color_idx,
                        patches=state.patches, super_areas=state.super_areas)
                state.edges_dirty = False

            if state.edges_dirty and state.merge_mode:
                state.edge_panel, state.inv_gray_cache = _make_edge_panel(
                    state.warped_display, state.sliders, state.color_idx,
                    patches=state.patches, super_areas=state.super_areas)
                state.edges_dirty = False

            if state.local_dirty:
                if state.local_mode and state.local_mask is not None:
                    state.local_edge_panel = _make_local_edge_panel(
                        state.warped_display, state.sliders, state.local_sliders,
                        state.local_mask, state.color_idx,
                        patches=state.patches, super_areas=state.super_areas)
                    if state.local_bbox is not None:
                        state.local_zoom_panel = _make_zoom_panel(
                            state.warped_display, state.local_bbox)
                state.local_dirty = False

            if state.merge_dirty and state.merge_mode:
                state.merge_edge_panel = _make_merge_edge_panel(
                    state.warped_display, state.sliders,
                    state.patches, state.super_areas,
                    state.color_idx, state.merge_active_sa_id)
                state.merge_dirty = False

            # Choose display panels
            if state.merge_mode and state.merge_edge_panel is not None:
                mid_panel   = state.merge_edge_panel
                right_panel = state.master_panel
            elif state.local_mode and state.local_edge_panel is not None:
                mid_panel   = state.local_edge_panel
                right_panel = (state.local_zoom_panel
                               if state.local_zoom_panel is not None
                               else state.master_panel)
            elif state.preview_take_idx is not None:
                entry = next((t for t in state.takes if t.index == state.preview_take_idx), None)
                if entry is not None:
                    mid_panel = _render_inv_gray(entry.display_inv_gray, state.color_idx)
                    # Right panel: selected color version or master
                    if state.preview_color_ver_id is not None:
                        cv_rec = next(
                            (c for c in entry.color_versions
                             if c.get("version_id") == state.preview_color_ver_id), None)
                        if cv_rec is not None and cv_rec.get("thumbnail") is not None:
                            try:
                                th = np.array(cv_rec["thumbnail"], dtype=np.uint8)
                                right_panel = _add_border(
                                    cv2.resize(th, (master_scaled.shape[1], master_scaled.shape[0]),
                                               interpolation=cv2.INTER_LINEAR), _BORDER)
                            except Exception:
                                right_panel = state.master_panel
                        else:
                            right_panel = state.master_panel
                    else:
                        right_panel = state.master_panel
                else:
                    mid_panel   = state.edge_panel
                    right_panel = state.master_panel
            elif state.overlay_on and state.overlay_panel is not None:
                mid_panel   = state.edge_panel
                right_panel = state.overlay_panel
            else:
                mid_panel   = state.edge_panel
                right_panel = state.master_panel

            ctrl_panel = _draw_ctrl_panel(
                state.sliders,
                sum(1 for t in state.takes if t.index > 0),
                state.color_idx,
                state.overlay_on,
                local_mode            = state.local_mode,
                local_sliders         = state.local_sliders or None,
                local_seal            = state.local_seal,
                patch_count           = len(state.patches),
                local_patch_idx       = state.local_patch_idx,
                merge_mode            = state.merge_mode,
                merge_sliders         = state.merge_sliders or None,
                merge_active_sa_id    = state.merge_active_sa_id,
                super_area_count      = len(state.super_areas),
                has_take_zero         = state.has_take_zero,
                print_preview_enabled = (state.preview_take_idx is not None),
            )

            target_h           = max(ctrl_panel.shape[0], mid_panel.shape[0], right_panel.shape[0])
            state.main_panel_h = target_h
            info_panel         = _draw_info_panel(state, target_h)

            main_row = np.hstack([
                _pad_to_height(ctrl_panel,  target_h),
                _pad_to_height(info_panel,  target_h),
                _pad_to_height(mid_panel,   target_h),
                _pad_to_height(right_panel, target_h),
            ])
            state.window_w = main_row.shape[1]
            film_panel = _make_filmstrip_panel(state, state.window_w)
            cv2.imshow(_EDGEMAP_WINDOW, np.vstack([main_row, film_panel]))

            # Key handling
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(_EDGEMAP_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                state.done = True

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
                elif state.merge_mode:
                    if state.merge_active_sa_id is not None:
                        _exit_sa_edit(state)
                        state.merge_dirty = True
                    else:
                        state.merge_mode       = False
                        state.merge_edge_panel = None
                        state.edges_dirty      = True
                elif state.preview_take_idx is not None:
                    if state.pre_seed_values is not None:
                        for sl, v in zip(state.sliders, state.pre_seed_values):
                            sl.value = v
                        state.pre_seed_values = None
                    state.preview_take_idx     = None
                    state.preview_color_ver_id = None
                    state.edges_dirty          = True
                else:
                    state.done = True

            elif key in (ord("m"), ord("M")):
                if not state.local_mode:
                    if state.merge_mode:
                        if state.merge_active_sa_id is not None:
                            _exit_sa_edit(state)
                        state.merge_mode       = False
                        state.merge_edge_panel = None
                        state.edges_dirty      = True
                    elif len(state.patches) >= 1 and state.has_take_zero:
                        state.merge_mode  = True
                        state.merge_dirty = True

            elif key in (ord("l"), ord("L")):
                if state.local_mode:
                    _commit_local_patch(state)
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

            # Handle done — check for print preview re-entry
            if state.done:
                if (state.print_preview_take_idx is not None
                        and out_dir is not None):
                    _run_print_preview(state, out_dir, stem or "output", jpg_quality)
                    state.print_preview_take_idx = None
                    state.done = False
                    continue
                break

    finally:
        try:
            cv2.destroyWindow(_EDGEMAP_WINDOW)
        except Exception:
            pass

    new_takes = [
        {
            "index":             t.index,
            "edges_full":        t.edges_full,
            "display_inv_gray":  t.display_inv_gray,
            "global_thresholds": t.global_thresholds,
            "local_info":        t.local_info,
            "seeded_from":       t.seeded_from,
            "base_image":        t.base_image,
            "patches_snapshot":  t.patches_snapshot,
        }
        for t in state.takes
        if t.is_new
    ]

    return (new_takes,
            _patches_to_session_data(state.patches),
            _super_areas_to_session_data(state.super_areas),
            state.colorize_take_idx)