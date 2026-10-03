#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confronta più run AlmenVision a partire dai CSV delle detection."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="CSV prodotti da detect_video.py",
    )
    return parser.parse_args()


def summarize(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    if df.empty:
        return {
            "run": csv_path.stem,
            "detections": 0,
            "unique_ids": 0,
            "avg_conf": 0.0,
            "min_conf": 0.0,
            "max_conf": 0.0,
            "frames_with_detections": 0,
            "max_per_frame": 0,
            "avg_per_frame": 0.0,
        }

    valid_ids = df["track_id"][df["track_id"] >= 0]

    detections_per_frame = df.groupby("frame").size()

    return {
        "run": csv_path.stem,
        "detections": len(df),
        "unique_ids": valid_ids.nunique(),
        "avg_conf": df["confidence"].mean(),
        "min_conf": df["confidence"].min(),
        "max_conf": df["confidence"].max(),
        "frames_with_detections": df["frame"].nunique(),
        "max_per_frame": detections_per_frame.max(),
        "avg_per_frame": detections_per_frame.mean(),
    }


def main() -> None:
    args = parse_args()

    results = []

    for csv_path in args.csv_files:
        if not csv_path.exists():
            print(f"ATTENZIONE: file non trovato: {csv_path}")
            continue

        results.append(summarize(csv_path))

    if not results:
        raise SystemExit("Nessun CSV valido da confrontare.")

    result_df = pd.DataFrame(results)

    result_df = result_df[
        [
            "run",
            "detections",
            "unique_ids",
            "frames_with_detections",
            "avg_conf",
            "min_conf",
            "max_conf",
            "avg_per_frame",
            "max_per_frame",
        ]
    ]

    print()
    print("AlmenVision - confronto run")
    print("=" * 110)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "avg_conf": lambda x: f"{x:.3f}",
                "min_conf": lambda x: f"{x:.3f}",
                "max_conf": lambda x: f"{x:.3f}",
                "avg_per_frame": lambda x: f"{x:.2f}",
            },
        )
    )

    print()
    print("Nota:")
    print("- 'detections' = box totali prodotte su tutti i frame")
    print("- 'unique_ids' = ID di tracking distinti")
    print("- 'frames_with_detections' = frame con almeno una mela rilevata")
    print("- 'avg_per_frame' = detection medie nei frame con almeno una detection")
    print("- 'max_per_frame' = massimo numero di mele rilevate nello stesso frame")


if __name__ == "__main__":
    main()
