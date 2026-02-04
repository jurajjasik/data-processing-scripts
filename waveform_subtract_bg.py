#!/usr/bin/env python3
"""
waveform_subtract_bg.py

Batch subtract a background waveform from CSV files with matching time points.

Input CSV format:
  time (seconds), value a, value b, ...
Example:
  time,a,b
  0.0,-0.8661417,-2.53937
  1.0048e-06,-0.629921,-2.53937
  ...

Rules:
- Ignores lines starting with '#'
- Background file given via --background
- For each input file: checks same number of rows and identical time points (within tolerance)
- Subtracts background values column-wise: out = data - background
- Writes output CSVs with suffix (default "-bgsub") and optional output directory
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from typing import List, Tuple, Optional


Row = Tuple[float, List[float]]


def parse_float(s: str) -> float:
    return float(s.strip())


def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_waveform_csv(path: Path) -> Tuple[List[str], List[Row]]:
    """
    Returns:
      fieldnames,
      rows: list of (time, [v1, v2, ...])
    Skips comment lines starting with '#'.
    """
    rows: List[Row] = []

    with path.open("r", newline="") as f:
        reader = csv.reader(f)

        header: Optional[List[str]] = None
        for raw in reader:
            if not raw:
                continue
            if raw[0].lstrip().startswith("#"):
                continue
            header = [c.strip() for c in raw]
            break

        if header is None or len(header) < 2:
            raise ValueError(f"{path}: Could not find a valid header (need time + >=1 value columns).")

        for raw in reader:
            if not raw:
                continue
            if raw[0].lstrip().startswith("#"):
                continue
            if len(raw) < len(header):
                continue

            try:
                t = parse_float(raw[0])
                vals = [parse_float(x) for x in raw[1:len(header)]]
            except ValueError:
                continue

            rows.append((t, vals))

    return header, rows


def expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in patterns:
        matched = glob.glob(p)
        if not matched and Path(p).is_file():
            matched = [p]
        files.extend(Path(m) for m in matched)

    # de-dupe preserving order
    seen = set()
    out: List[Path] = []
    for f in files:
        rp = str(f.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def build_output_path(in_path: Path, out_dir: Optional[Path], suffix: str, out_ext: str) -> Path:
    out_name = f"{in_path.stem}{suffix}{out_ext}"
    return (out_dir / out_name) if out_dir else in_path.with_name(out_name)


def times_match(a: float, b: float, atol: float, rtol: float) -> bool:
    # like numpy.isclose
    return abs(a - b) <= (atol + rtol * abs(b))


def subtract_background(
    data: List[Row],
    bg: List[Row],
    *,
    time_atol: float,
    time_rtol: float,
    allow_time_sort: bool,
) -> List[Row]:
    if len(data) != len(bg):
        raise ValueError(f"Row count mismatch: data={len(data)} bg={len(bg)}")

    if not data:
        return []

    # Optionally sort both by time (in case files are written unsorted)
    if allow_time_sort:
        data = sorted(data, key=lambda x: x[0])
        bg = sorted(bg, key=lambda x: x[0])

    ncols = len(data[0][1])
    if any(len(v) != ncols for _, v in data):
        raise ValueError("Data file has inconsistent number of value columns across rows.")
    if any(len(v) != ncols for _, v in bg):
        raise ValueError("Background file column count does not match data file (or inconsistent rows).")

    out: List[Row] = []
    for i, ((t, dv), (tb, bv)) in enumerate(zip(data, bg)):
        if not times_match(t, tb, time_atol, time_rtol):
            raise ValueError(
                f"Time mismatch at row {i}: data t={t} bg t={tb} "
                f"(atol={time_atol}, rtol={time_rtol})"
            )
        out_vals = [dv[j] - bv[j] for j in range(ncols)]
        out.append((t, out_vals))

    return out


def write_waveform_csv(path: Path, fieldnames: List[str], rows: List[Row]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for t, vals in rows:
            writer.writerow([t, *vals])


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch subtract a background waveform from CSV files.")
    ap.add_argument(
        "inputs",
        nargs="+",
        help='Input files or glob patterns, e.g. "*.csv" data/*.csv',
    )
    ap.add_argument(
        "--background",
        "-b",
        required=True,
        help="Background CSV file to subtract (same time points required).",
    )
    ap.add_argument(
        "--suffix",
        default="-bgsub",
        help='Output filename suffix before extension (default: "-bgsub")',
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory (created if missing). If omitted, writes next to inputs.",
    )
    ap.add_argument(
        "--ext",
        default=".csv",
        help='Output extension (default: ".csv")',
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    ap.add_argument(
        "--time-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for time equality check (default: 0.0).",
    )
    ap.add_argument(
        "--time-rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for time equality check (default: 0.0).",
    )
    ap.add_argument(
        "--allow-time-sort",
        action="store_true",
        help="Sort by time before comparison/subtraction (helps if rows are not ordered).",
    )
    ap.add_argument(
        "--skip-background-input",
        action="store_true",
        help="If the background file is among the inputs, skip processing it.",
    )

    args = ap.parse_args()

    bg_path = Path(args.background)
    if not bg_path.is_file():
        print(f"ERROR: background file not found: {bg_path}")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        safe_makedirs(out_dir)

    # Read background
    try:
        bg_header, bg_rows = read_waveform_csv(bg_path)
    except Exception as e:
        print(f"ERROR reading background {bg_path}: {e}")
        return 2

    if len(bg_header) < 2:
        print(f"ERROR: background {bg_path} has invalid header.")
        return 2

    in_files = expand_inputs(args.inputs)
    if not in_files:
        print("No input files matched.")
        return 2

    processed = 0
    skipped = 0

    for in_path in in_files:
        try:
            if args.skip_background_input and in_path.resolve() == bg_path.resolve():
                print(f"SKIP (background): {in_path}")
                skipped += 1
                continue

            header, rows = read_waveform_csv(in_path)

            # Basic header compatibility checks:
            if len(header) != len(bg_header):
                raise ValueError(
                    f"Header column count mismatch vs background: data={len(header)} bg={len(bg_header)}"
                )
            # If you want strict header name match, uncomment:
            # if [h.strip() for h in header] != [h.strip() for h in bg_header]:
            #     raise ValueError("Header names mismatch vs background.")

            out_rows = subtract_background(
                rows,
                bg_rows,
                time_atol=float(args.time_atol),
                time_rtol=float(args.time_rtol),
                allow_time_sort=bool(args.allow_time_sort),
            )

            out_path = build_output_path(in_path, out_dir, str(args.suffix), str(args.ext))
            if out_path.exists() and not args.overwrite:
                print(f"SKIP (exists): {out_path}")
                skipped += 1
                continue

            write_waveform_csv(out_path, header, out_rows)
            print(f"OK: {in_path} -> {out_path} | rows={len(out_rows)}")
            processed += 1

        except Exception as e:
            print(f"ERROR: {in_path}: {e}")
            skipped += 1

    print(f"Done. Processed={processed}, Skipped/Failed={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
