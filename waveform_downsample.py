#!/usr/bin/env python3
"""
waveform_downsample.py

Batch downsample waveform CSV files by averaging points within fixed time bins.

Input CSV format:
  time (seconds), value a, value b
Example:
  time,a,b
  0.0,-0.8661417,-2.53937
  1.0048e-06,-0.629921,-2.53937
  ...

Features:
- Accepts wildcard/glob patterns (e.g., "*.csv")
- Ignores lines starting with '#'
- Downsamples by time-binning with width dt (seconds), computing mean time and mean of each value column per bin
- Writes output CSVs with suffix (default "-ds") and optional output directory
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional


def parse_float(s: str) -> float:
    # Handles scientific notation, trims whitespace
    return float(s.strip())


def safe_makedirs(path: Path) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)


def read_waveform_csv(path: Path) -> Tuple[str, List[str], List[Tuple[float, List[float]]]]:
    """
    Returns:
      header_line (raw, joined for reference),
      fieldnames (parsed),
      rows: list of (time, [v1, v2, ...])
    Skips lines starting with '#'.
    """
    rows: List[Tuple[float, List[float]]] = []

    with path.open("r", newline="") as f:
        # Skip comment lines until we hit a non-comment line (likely header)
        header_row: Optional[List[str]] = None
        reader = csv.reader(f)

        for raw in reader:
            if not raw:
                continue
            first_cell = raw[0].lstrip()
            if first_cell.startswith("#"):
                continue
            header_row = [c.strip() for c in raw]
            break

        if header_row is None or len(header_row) < 2:
            raise ValueError(f"{path}: Could not find a valid header row (needs at least time + 1 value column).")

        fieldnames = header_row
        header_line = ",".join(fieldnames)

        # Read data rows
        for raw in reader:
            if not raw:
                continue
            first_cell = raw[0].lstrip()
            if first_cell.startswith("#"):
                continue

            # Allow extra whitespace; require at least same number of columns as header
            if len(raw) < len(fieldnames):
                # Skip malformed short lines
                continue

            try:
                t = parse_float(raw[0])
                vals = [parse_float(x) for x in raw[1:len(fieldnames)]]
            except ValueError:
                # Skip lines that don't parse
                continue

            rows.append((t, vals))

    return header_line, fieldnames, rows


def downsample_time_bins(
    rows: List[Tuple[float, List[float]]],
    dt: float,
) -> List[Tuple[float, List[float]]]:
    """
    Downsample by grouping samples into fixed-width bins in time:
      bin_index = floor(t / dt)
    For each bin: output mean time and mean for each value column.

    Assumes times are non-decreasing; if not, we still bin by absolute t/dt.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")

    if not rows:
        return []

    # Sort by time to make bin transitions sane (cheap insurance)
    rows_sorted = sorted(rows, key=lambda x: x[0])

    out: List[Tuple[float, List[float]]] = []

    current_bin: Optional[int] = None
    sum_t = 0.0
    sum_vals: List[float] = []
    n = 0

    for t, vals in rows_sorted:
        b = int(math.floor(t / dt))

        if current_bin is None:
            current_bin = b
            sum_t = t
            sum_vals = vals[:]  # copy
            n = 1
            continue

        if b == current_bin:
            sum_t += t
            for i, v in enumerate(vals):
                sum_vals[i] += v
            n += 1
        else:
            # flush previous bin
            mean_t = sum_t / n
            mean_vals = [sv / n for sv in sum_vals]
            out.append((mean_t, mean_vals))

            # start new bin
            current_bin = b
            sum_t = t
            sum_vals = vals[:]
            n = 1

    # flush last bin
    if current_bin is not None and n > 0:
        mean_t = sum_t / n
        mean_vals = [sv / n for sv in sum_vals]
        out.append((mean_t, mean_vals))

    return out


def build_output_path(
    in_path: Path,
    out_dir: Optional[Path],
    suffix: str,
    out_ext: str,
) -> Path:
    """
    Example:
      input:  data/run1.csv
      suffix: -ds
      out:    data/run1-ds.csv  (or out_dir/run1-ds.csv)
    """
    stem = in_path.stem  # "run1" (for run1.csv)
    out_name = f"{stem}{suffix}{out_ext}"
    if out_dir is None:
        return in_path.with_name(out_name)
    return out_dir / out_name


def write_waveform_csv(
    path: Path,
    fieldnames: List[str],
    rows: List[Tuple[float, List[float]]],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for t, vals in rows:
            writer.writerow([t, *vals])


def expand_inputs(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in patterns:
        matched = glob.glob(p)
        if not matched:
            # also allow a literal file path with no glob match
            if Path(p).is_file():
                matched = [p]
        files.extend(Path(m) for m in matched)
    # de-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for f in files:
        fp = str(f.resolve())
        if fp not in seen:
            seen.add(fp)
            out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch downsample waveform CSV files by averaging within time bins."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help='Input files or glob patterns, e.g. "*.csv" data/*.csv',
    )
    ap.add_argument(
        "--dt",
        required=True,
        type=float,
        help="Downsample bin width in seconds, e.g. 1e-4",
    )
    ap.add_argument(
        "--suffix",
        default="-ds",
        help='Output filename suffix before extension (default: "-ds")',
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

    args = ap.parse_args()

    dt = float(args.dt)
    suffix = str(args.suffix)
    out_ext = str(args.ext)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if out_dir is not None:
        safe_makedirs(out_dir)

    in_files = expand_inputs(args.inputs)
    if not in_files:
        print("No input files matched.")
        return 2

    processed = 0
    skipped = 0

    for in_path in in_files:
        try:
            header_line, fieldnames, rows = read_waveform_csv(in_path)
            if len(fieldnames) < 2:
                raise ValueError("Need at least time + one value column.")
            ds_rows = downsample_time_bins(rows, dt)

            out_path = build_output_path(in_path, out_dir, suffix, out_ext)
            if out_path.exists() and not args.overwrite:
                print(f"SKIP (exists): {out_path}")
                skipped += 1
                continue

            write_waveform_csv(out_path, fieldnames, ds_rows)
            print(
                f"OK: {in_path} -> {out_path} | "
                f"in={len(rows)} rows, out={len(ds_rows)} rows, dt={dt}"
            )
            processed += 1
        except Exception as e:
            print(f"ERROR: {in_path}: {e}")
            skipped += 1

    print(f"Done. Processed={processed}, Skipped/Failed={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
