from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# EXIF tag IDs we probe, in preference order.
# 0x9003 = DateTimeOriginal, 0x9004 = DateTimeDigitized, 0x0132 = DateTime
_EXIF_DATETIME_TAGS = (0x9003, 0x9004, 0x0132)
_EXIF_DATETIME_FMT  = "%Y:%m:%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_jpg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError(f"Could not write image: {path}")


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def iter_input_images(path: Path) -> Iterable[Path]:
    """Yield image files under *path* (single file or directory)."""
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return

    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def iter_master_images(path: Path) -> Iterable[Path]:
    """Yield ``*_master.jpg`` files under *path* (single file or directory).

    A single file is accepted only if its stem ends with ``_master``.
    A directory is scanned for all files whose stem ends with ``_master``
    and whose extension is a recognised image type, sorted alphabetically.
    """
    if path.is_file():
        if path.stem.endswith("_master") and path.suffix.lower() in IMAGE_EXTS:
            yield path
        return

    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.is_file() and p.stem.endswith("_master") and p.suffix.lower() in IMAGE_EXTS:
                yield p


def stem_from_master(master_path: Path) -> str:
    """Derive the output stem from a ``*_master.jpg`` path.

    ``SA221a_master.jpg``  →  ``"SA221a"``
    """
    stem = master_path.stem
    if stem.endswith("_master"):
        return stem[: -len("_master")]
    return stem


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# EXIF / datetime helpers
# ---------------------------------------------------------------------------

def read_capture_datetime(path: Path) -> datetime:
    """
    Return the capture datetime for *path*.

    Strategy (in order of preference):
    1. EXIF DateTimeOriginal / DateTimeDigitized / DateTime tags embedded in
       the JPEG — most accurate, reflects when the shutter fired.
    2. File modification time — used when no EXIF data is present (e.g. the
       file has been stripped or is a non-JPEG format).

    The returned value is a naive ``datetime`` in local time.
    """
    # Try EXIF first (JPEG / TIFF only; silently skip for other formats)
    if path.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}:
        try:
            import piexif
            exif = piexif.load(str(path))
            for ifd in ("Exif", "0th", "1st"):
                ifd_data = exif.get(ifd, {})
                for tag in _EXIF_DATETIME_TAGS:
                    raw = ifd_data.get(tag)
                    if raw:
                        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
                        text = text.strip().rstrip("\x00")
                        try:
                            return datetime.strptime(text, _EXIF_DATETIME_FMT)
                        except ValueError:
                            pass
        except Exception:
            pass  # piexif not installed, or malformed EXIF — fall through

    # Fall back to filesystem mtime
    return datetime.fromtimestamp(path.stat().st_mtime)


def sort_images_by_datetime(paths: list[Path]) -> list[tuple[Path, datetime]]:
    """
    Return *paths* sorted earliest-first by capture datetime.
    Each element of the returned list is ``(path, datetime)``.
    """
    tagged = [(p, read_capture_datetime(p)) for p in paths]
    tagged.sort(key=lambda t: t[1])
    return tagged


# ---------------------------------------------------------------------------
# Batch name generation
# ---------------------------------------------------------------------------

def generate_batch_names(
    n_images: int,
    prefix: str,
    start_num: int,
    start_letter: str = "a",
) -> list[str]:
    """
    Generate *n_images* sequential output stems.

    Names take the form ``{prefix}{start_num}{letter}`` where *letter* advances
    from *start_letter* through ``z``.  Example: ``SA221a``, ``SA221b``, …

    Parameters
    ----------
    n_images:
        How many names to generate.
    prefix:
        Alphabetic prefix, e.g. ``"SA"``.
    start_num:
        Numeric part of the stem, e.g. ``221``.
    start_letter:
        Letter to begin from (``'a'``–``'z'``, case-insensitive).

    Raises
    ------
    ValueError
        If *start_letter* is not a lowercase ASCII letter, or if *n_images*
        would require letters beyond ``'z'``.
    """
    start_letter = start_letter.lower()
    if start_letter not in string.ascii_lowercase:
        raise ValueError(f"start_letter must be a–z, got {start_letter!r}")

    start_idx  = string.ascii_lowercase.index(start_letter)
    slots_left = len(string.ascii_lowercase) - start_idx

    if n_images > slots_left:
        raise ValueError(
            f"Cannot generate {n_images} names starting from {start_letter!r}: "
            f"only {slots_left} letter(s) available ('{start_letter}'–'z'). "
            f"Reduce the number of images or choose an earlier start_letter."
        )

    return [
        f"{prefix}{start_num}{string.ascii_lowercase[start_idx + i]}"
        for i in range(n_images)
    ]


# ---------------------------------------------------------------------------
# Image geometry helpers
# ---------------------------------------------------------------------------

def resize_for_preview(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w    = image.shape[:2]
    largest = max(h, w)

    if largest <= max_dim:
        return image.copy(), 1.0

    scale  = max_dim / float(largest)
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_points(points: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return points.astype(np.float32) / scale


def build_output_path(src_path: Path, out_dir: Path, suffix: str = "_master") -> Path:
    return out_dir / f"{src_path.stem}{suffix}.jpg"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners ordered [top-left, top-right, bottom-right, bottom-left]."""
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2) points, got {pts.shape}")

    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left  (smallest x+y)
    rect[2] = pts[np.argmax(s)]   # bottom-right (largest x+y)

    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right  (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # bottom-left (largest y-x)

    return rect


def draw_quad(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            out, str(i), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA,
        )
    return out

 
# ---------------------------------------------------------------------------
# Lab-based multi-channel edge detection
# ---------------------------------------------------------------------------
 
def compute_lab_edges(
    image: np.ndarray,
    l_lo: int,
    l_hi: int,
    a_lo: int,
    a_hi: int,
    b_lo: int,
    b_hi: int,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Compute a merged edge map from all three CIE Lab channels via Canny and
    return an *inverted* grayscale image (white background, black edges).
 
    Running Canny independently on L*, a*, and b* captures edges that are
    invisible to luminance-only detection — most notably yellow strokes on
    light grounds, which vanish in L* but are strong in b*.
 
    Parameters
    ----------
    image:
        BGR input image (any resolution).
    l_lo, l_hi:
        Canny hysteresis thresholds for the L* (luminance) channel.
        Pass l_hi=0 to suppress the luminance channel entirely.
    a_lo, a_hi:
        Canny thresholds for the a* (green↔red) channel.
        Pass a_hi=0 to suppress.
    b_lo, b_hi:
        Canny thresholds for the b* (blue↔yellow) channel.
        Pass b_hi=0 to suppress.
    blur_ksize:
        Gaussian blur kernel size applied to each channel before Canny.
        Must be odd and ≥ 1.
 
    Returns
    -------
    np.ndarray
        Grayscale image (dtype uint8): 255 = background, 0 = edge.
    """
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    ksize = max(1, ksize)
 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)
 
    def _canny(ch: np.ndarray, lo: int, hi: int) -> np.ndarray:
        if hi <= 0:
            return np.zeros(ch.shape, dtype=np.uint8)
        blurred = cv2.GaussianBlur(ch, (ksize, ksize), 0)
        return cv2.Canny(blurred, lo, hi)
 
    edges = np.maximum(
        np.maximum(_canny(l_ch, l_lo, l_hi), _canny(a_ch, a_lo, a_hi)),
        _canny(b_ch, b_lo, b_hi),
    )
    return cv2.bitwise_not(edges)

# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1


@dataclass
class SessionData:
    """Complete record of one image's processing session.

    Stored as ``{stem}_session.json`` in the output directory.  Written only
    after a successful acceptance (corner edit + optional edge-map takes) so
    that a missing file unambiguously means "not yet processed".
    """
    schema_version:     int
    timestamp:          str           # ISO-8601, local time
    source_path:        str           # absolute path to the original photo
    output_stem:        str           # e.g. "SA221a"
    corners_full:       list          # [[x,y]×4] in full-resolution pixels
    initial_thresholds: dict          # l_lo/l_hi/a_lo/a_hi/b_lo/b_hi seed values
    takes:              list          # one dict per Take, keys: index + threshold names
    local_regions:      list = field(default_factory=list)   # reserved


def _thresholds_dict(
    l_lo: int, l_hi: int,
    a_lo: int, a_hi: int,
    b_lo: int, b_hi: int,
) -> dict:
    return {
        "l_lo": l_lo, "l_hi": l_hi,
        "a_lo": a_lo, "a_hi": a_hi,
        "b_lo": b_lo, "b_hi": b_hi,
    }


def save_session(session_path: Path, data: SessionData) -> None:
    """Serialise *data* to *session_path* as pretty-printed JSON."""
    payload = {
        "schema_version":     data.schema_version,
        "timestamp":          data.timestamp,
        "source_path":        data.source_path,
        "output_stem":        data.output_stem,
        "corners_full":       data.corners_full,
        "initial_thresholds": data.initial_thresholds,
        "takes":              data.takes,
        "local_regions":      data.local_regions,
    }
    session_path.parent.mkdir(parents=True, exist_ok=True)
    with open(session_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def load_session(session_path: Path) -> SessionData | None:
    """Load a session JSON from *session_path*.

    Returns ``None`` — without raising — if the file does not exist or cannot
    be parsed, so callers can treat a missing/corrupt file as "no session".
    """
    if not session_path.exists():
        return None
    try:
        with open(session_path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return SessionData(
            schema_version     = int(d.get("schema_version", 1)),
            timestamp          = str(d.get("timestamp", "")),
            source_path        = str(d.get("source_path", "")),
            output_stem        = str(d.get("output_stem", "")),
            corners_full       = d.get("corners_full", []),
            initial_thresholds = d.get("initial_thresholds", {}),
            takes              = d.get("takes", []),
            local_regions      = d.get("local_regions", []),
        )
    except Exception:
        return None


def session_path_for(out_dir: Path, stem: str) -> Path:
    """Canonical location of a session file given output directory and stem."""
    return out_dir / f"{stem}_session.json"


def thresholds_from_session(
    session: SessionData,
    *,
    prefer_last_take: bool = True,
) -> dict:
    """Extract the most useful threshold dict from a loaded session.

    When *prefer_last_take* is True (the default) and at least one Take was
    recorded, return the last Take's thresholds — this is where the user
    ended up and is the best seed for a re-entry session.  Fall back to
    ``initial_thresholds`` if there are no Takes.
    """
    if prefer_last_take and session.takes:
        return dict(session.takes[-1])   # shallow copy; caller may mutate
    return dict(session.initial_thresholds)


def draw_quad_print(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Like draw_quad but scaled for print output (thick lines, large labels)."""
    out = image.copy()
    long_side = max(out.shape[:2])
    thickness = max(4, long_side // 300)
    radius    = max(12, long_side // 200)
    font_scale = max(1.0, long_side / 1200)

    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    for i, (x, y) in enumerate(corners.astype(np.int32)):
        cv2.circle(out, (x, y), radius, (0, 0, 255), -1)
        cv2.putText(
            out, str(i), (x + radius + 4, y - radius),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
            max(2, thickness - 1), cv2.LINE_AA,
        )
    return out