from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanner import ProcessResult, ScanConfig, process_image
from utils import (
    ensure_out_dir,
    generate_batch_names,
    iter_input_images,
    sort_images_by_datetime,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and perspective-correct paintings from phone photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single image, interactive editor:
  python main.py photo.jpg --out output/

  # Whole folder, auto-detect only:
  python main.py data/ --out output/ --no-interactive

  # Batch with renaming: sort data/ by capture time, name SA221a, SA221b, …
  python main.py data/ --out output/ --prefix SA --num 221 --letter a

  # Start partway through the alphabet (e.g. continuing a previous run):
  python main.py data/ --out output/ --prefix SA --num 221 --letter f

  # Output relative to the input's directory (not the CWD):
  python main.py c:/data/paintscan/data/SA223/SA223.jpg --out ./run4f --edgemap

  # Generate Lab edge maps interactively after each warp:
  python main.py data/ --out output/ --edgemap
""",
    )

    # --- positional ---
    parser.add_argument(
        "input",
        type=str,
        help="Input image file or folder.",
    )

    # --- output ---
    parser.add_argument(
        "--out",
        type=str,
        default="output",
        help="Output folder (default: output). Prefix with ./ to resolve relative to the input's directory.",
    )

    # --- batch rename ---
    rename = parser.add_argument_group(
        "batch renaming",
        "When --prefix and --num are both given the images are sorted by capture\n"
        "datetime and named {prefix}{num}{letter}, e.g. SA221a, SA221b, …\n"
        "Letters advance from --letter (default: a) through z.",
    )
    rename.add_argument(
        "--prefix",
        type=str,
        default=None,
        metavar="PREFIX",
        help='Alphabetic prefix for output stems, e.g. "SA".',
    )
    rename.add_argument(
        "--num",
        type=int,
        default=None,
        metavar="NUM",
        help="Numeric part of the output stem, e.g. 221.",
    )
    rename.add_argument(
        "--letter",
        type=str,
        default="a",
        metavar="LETTER",
        help="Starting letter (a–z, default: a).",
    )

    # --- scanner tuning ---
    tune = parser.add_argument_group("scanner tuning")
    tune.add_argument(
        "--max-dim", type=int, default=1600,
        help="Max preview dimension for detection (default: 1600).",
    )
    tune.add_argument(
        "--blur", type=int, default=5,
        help="Gaussian blur kernel size (default: 5).",
    )
    tune.add_argument(
        "--canny-lo", type=int, default=50,
        help="Canny low threshold (default: 50).",
    )
    tune.add_argument(
        "--canny-hi", type=int, default=150,
        help="Canny high threshold (default: 150).",
    )
    tune.add_argument(
        "--min-area-ratio", type=float, default=0.10,
        help="Min contour area as fraction of image (default: 0.10).",
    )
    tune.add_argument(
        "--trim", type=int, default=2,
        help="Pixels to trim from each edge after warp (default: 2).",
    )
    tune.add_argument(
        "--quality", type=int, default=95,
        help="JPEG output quality 1–100 (default: 95).",
    )

    # --- flags ---
    parser.add_argument(
        "--debug", action="store_true",
        help="Write debug overlay, edge-map and Hough-line images.",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip the corner editor; use automatic detection only.",
    )
    parser.add_argument(
        "--edgemap", action="store_true",
        help=(
            "After each warp, open an interactive Lab edge-map tuner.  "
            "Saves {stem}_edges.jpg (full resolution) and {stem}_edges_600.jpg "
            "alongside the master outputs.  Sliders control Canny thresholds "
            "for the L*, a*, and b* channels independently."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_rename_args(args: argparse.Namespace) -> None:
    """Abort early with a clear message if rename args are partially specified."""
    has_prefix = args.prefix is not None
    has_num    = args.num is not None
    if has_prefix != has_num:
        sys.exit(
            "error: --prefix and --num must be used together.\n"
            "  Example: --prefix SA --num 221"
        )
    if args.letter and len(args.letter) != 1 or (args.letter and args.letter not in "abcdefghijklmnopqrstuvwxyz"):
        sys.exit(f"error: --letter must be a single lowercase letter a–z, got {args.letter!r}")


def _print_result(res: ProcessResult, output_stem: str | None) -> None:
    """Print a one-line summary for a processed image."""
    label = res.output_master.name if res.output_master else "?"
    src   = res.source.name

    if res.accepted:
        method_info = (
            f"method={res.method}, interactive"
            if res.used_interactive
            else f"method={res.method}, score={res.confidence:.3f}"
            if res.confidence is not None
            else f"method={res.method}"
        )
        stem_info    = f" [{output_stem}]" if output_stem else ""
        session_info = f"  session={res.session_path.name}" if res.session_path else ""
        print(f"[OK]   {src} -> {label}  ({method_info}){stem_info}{session_info}")
    else:
        print(f"[FAIL] {src} -> {res.message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    _validate_rename_args(args)

    input_path = Path(args.input)
    # Allow "--out ./subdir" to mean: relative to the input's parent directory
    # (or the input itself when it is a directory) rather than the CWD.
    raw_out = args.out
    if raw_out.startswith("./") or raw_out.startswith(".\\"):
        anchor  = input_path if input_path.is_dir() else input_path.parent
        out_dir = ensure_out_dir(anchor / raw_out[2:])
    else:
        out_dir = ensure_out_dir(Path(raw_out))

    cfg = ScanConfig(
        downscale_max_dim=args.max_dim,
        blur_ksize=args.blur,
        canny_lo=args.canny_lo,
        canny_hi=args.canny_hi,
        contour_min_area_ratio=args.min_area_ratio,
        jpg_quality=args.quality,
        trim_px=args.trim,
        debug=args.debug,
        interactive=not args.no_interactive,
        edgemap=args.edgemap,
    )

    # --- collect images ---
    raw_images = list(iter_input_images(input_path))
    if not raw_images:
        print("No input images found.")
        return 1

    # --- batch renaming: sort by datetime and assign stems ---
    use_rename = (args.prefix is not None and args.num is not None)

    if use_rename:
        try:
            sorted_pairs = sort_images_by_datetime(raw_images)
        except Exception as exc:
            sys.exit(f"error reading image datetimes: {exc}")

        try:
            names = generate_batch_names(
                n_images=len(sorted_pairs),
                prefix=args.prefix,
                start_num=args.num,
                start_letter=args.letter,
            )
        except ValueError as exc:
            sys.exit(f"error: {exc}")

        work_items: list[tuple[Path, str | None]] = [
            (path, stem) for (path, _dt), stem in zip(sorted_pairs, names)
        ]

        print(f"[INFO] Batch rename: {len(work_items)} image(s), "
              f"{args.prefix}{args.num}{args.letter} … "
              f"{names[-1]}")
        print("[INFO] Capture-datetime order:")
        for (path, dt), stem in zip(sorted_pairs, names):
            print(f"         {stem}  {path.name}  ({dt:%Y-%m-%d %H:%M:%S})")
        print()
    else:
        work_items = [(p, None) for p in raw_images]

    # --- process ---
    ok_count   = 0
    fail_count = 0

    for img_path, output_stem in work_items:
        print(f"[INFO] Processing: {img_path.name}"
              + (f"  ->  {output_stem}" if output_stem else ""))
        res = process_image(img_path, out_dir, cfg, output_stem=output_stem)
        _print_result(res, output_stem)

        if res.accepted:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nDone.  OK={ok_count}  FAIL={fail_count}")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())