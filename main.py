from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scanner import ProcessResult, ScanConfig, process_image, process_from_master
from utils import (
    ensure_out_dir,
    generate_batch_names,
    iter_input_images,
    iter_master_images,
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

  # Re-enter Lab editor for a single already-warped master (session restored):
  python main.py SA221a_master.jpg --from-master

  # Re-enter Lab editor for all masters in an output folder:
  python main.py output/ --from-master
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
        default=None,
        help=(
            "Output folder.  Defaults to 'output' in normal mode, or to the "
            "master file's parent directory in --from-master mode.  "
            "Prefix with ./ to resolve relative to the input's directory."
        ),
    )

    # --- re-entry mode ---
    parser.add_argument(
        "--from-master",
        action="store_true",
        help=(
            "Re-enter the Lab edge-map editor for an already-warped "
            "*_master.jpg (or a folder of them).  Skips detection and corner "
            "editing entirely.  Session thresholds are restored from the "
            "matching *_session.json when present."
        ),
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
            if res.used_interactive and res.method != "from-master"
            else f"method={res.method}, score={res.confidence:.3f}"
            if res.confidence is not None
            else f"method={res.method}"
        )
        stem_info    = f" [{output_stem}]" if output_stem else ""
        session_info = f"  session={res.session_path.name}" if res.session_path else ""
        print(f"[OK]   {src} -> {label}  ({method_info}){stem_info}{session_info}")
    else:
        print(f"[FAIL] {src} -> {res.message}")


def _resolve_out_dir(args: argparse.Namespace, input_path: Path) -> Path:
    """Return the resolved output directory for normal (non-from-master) mode."""
    raw_out = args.out or "output"
    if raw_out.startswith("./") or raw_out.startswith(".\\"):
        anchor = input_path if input_path.is_dir() else input_path.parent
        return ensure_out_dir(anchor / raw_out[2:])
    return ensure_out_dir(Path(raw_out))


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_normal(args: argparse.Namespace) -> int:
    """Standard detect → corner-edit → warp → (edgemap) pipeline."""
    _validate_rename_args(args)

    input_path = Path(args.input)
    out_dir    = _resolve_out_dir(args, input_path)

    cfg = ScanConfig(
        downscale_max_dim      = args.max_dim,
        blur_ksize             = args.blur,
        canny_lo               = args.canny_lo,
        canny_hi               = args.canny_hi,
        contour_min_area_ratio = args.min_area_ratio,
        jpg_quality            = args.quality,
        trim_px                = args.trim,
        debug                  = args.debug,
        interactive            = not args.no_interactive,
        edgemap                = args.edgemap,
    )

    raw_images = list(iter_input_images(input_path))
    if not raw_images:
        print("No input images found.")
        return 1

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
              f"{args.prefix}{args.num}{args.letter} … {names[-1]}")
        print("[INFO] Capture-datetime order:")
        for (path, dt), stem in zip(sorted_pairs, names):
            print(f"         {stem}  {path.name}  ({dt:%Y-%m-%d %H:%M:%S})")
        print()
    else:
        work_items = [(p, None) for p in raw_images]

    ok_count = fail_count = 0
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


def run_from_master(args: argparse.Namespace) -> int:
    """Re-entry mode: load *_master.jpg, restore session, open Lab editor."""
    input_path = Path(args.input)

    masters = list(iter_master_images(input_path))
    if not masters:
        print(f"No *_master.jpg files found at: {input_path}")
        return 1

    cfg = ScanConfig(
        jpg_quality = args.quality,
        canny_lo    = args.canny_lo,
        canny_hi    = args.canny_hi,
        debug       = args.debug,
        # lab a/b defaults — overridden by session restore when available
        lab_a_lo    = 30,
        lab_a_hi    = 90,
        lab_b_lo    = 30,
        lab_b_hi    = 90,
    )

    print(f"[INFO] From-master mode: {len(masters)} file(s)")

    ok_count = fail_count = 0
    for master_path in masters:
        # Out dir: explicit --out wins; otherwise use the master's own directory
        if args.out is not None:
            raw_out = args.out
            if raw_out.startswith("./") or raw_out.startswith(".\\"):
                anchor  = master_path.parent
                out_dir = ensure_out_dir(anchor / raw_out[2:])
            else:
                out_dir = ensure_out_dir(Path(raw_out))
        else:
            out_dir = master_path.parent  # outputs already live here

        print(f"[INFO] Re-entering: {master_path.name}  (out: {out_dir})")
        res = process_from_master(master_path, out_dir, cfg)
        _print_result(res, output_stem=None)
        if res.accepted:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nDone.  OK={ok_count}  FAIL={fail_count}")
    return 0 if ok_count > 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.from_master:
        return run_from_master(args)
    return run_normal(args)


if __name__ == "__main__":
    raise SystemExit(main())