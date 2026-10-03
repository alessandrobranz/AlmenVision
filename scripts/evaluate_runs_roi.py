#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Valuta detection AlmenVision solo dentro la ROI operativa."
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/ground_truth_roi.json"),
    )
    p.add_argument("csv_files", nargs="+", type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation_roi"),
    )
    return p.parse_args()


def inside_roi(x: float, y: float, roi: list[list[int]]) -> bool:
    if len(roi) < 3:
        return True
    contour = np.array(roi, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def point_inside_box(point, row) -> bool:
    x, y = point
    return row.x1 <= x <= row.x2 and row.y1 <= y <= row.y2


def match_frame(gt_points, detections):
    if detections.empty:
        return 0, 0, len(gt_points)

    detections = detections.sort_values("confidence", ascending=False)
    unmatched = set(range(len(gt_points)))
    tp = 0
    fp = 0

    for row in detections.itertuples(index=False):
        candidates = [
            i for i in unmatched
            if point_inside_box(gt_points[i], row)
        ]
        if not candidates:
            fp += 1
            continue

        cx = (row.x1 + row.x2) / 2
        cy = (row.y1 + row.y2) / 2
        best = min(
            candidates,
            key=lambda i: (gt_points[i][0] - cx) ** 2 + (gt_points[i][1] - cy) ** 2,
        )
        unmatched.remove(best)
        tp += 1

    return tp, fp, len(unmatched)


def safe_div(a, b):
    return a / b if b else 0.0


def evaluate(csv_path: Path, gt_frames: dict):
    df = pd.read_csv(csv_path)
    rows = []

    for frame_key, info in gt_frames.items():
        frame = int(frame_key)
        roi = info.get("roi", [])
        gt_points = [tuple(map(float, p)) for p in info.get("apples", [])]

        det = df[df["frame"] == frame].copy()

        if not det.empty and len(roi) >= 3:
            centers_x = (det["x1"] + det["x2"]) / 2
            centers_y = (det["y1"] + det["y2"]) / 2
            mask = [
                inside_roi(x, y, roi)
                for x, y in zip(centers_x, centers_y)
            ]
            det = det[mask]

        tp, fp, fn = match_frame(gt_points, det)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append({
            "frame": frame,
            "gt_apples": len(gt_points),
            "detections_in_roi": len(det),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    per_frame = pd.DataFrame(rows)

    tp = int(per_frame.tp.sum())
    fp = int(per_frame.fp.sum())
    fn = int(per_frame.fn.sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "run": csv_path.stem,
        "frames_evaluated": len(per_frame),
        "gt_apples": int(per_frame.gt_apples.sum()),
        "detections_in_roi": int(per_frame.detections_in_roi.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return summary, per_frame


def main():
    args = parse_args()

    with args.ground_truth.open("r", encoding="utf-8") as fh:
        gt = json.load(fh)

    gt_frames = gt.get("frames", {})
    if not gt_frames:
        raise SystemExit("Ground truth ROI vuota.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for csv_path in args.csv_files:
        if not csv_path.exists():
            print(f"ATTENZIONE: non trovato {csv_path}")
            continue

        summary, per_frame = evaluate(csv_path, gt_frames)
        summaries.append(summary)
        per_frame.to_csv(
            args.output_dir / f"{csv_path.stem}_per_frame.csv",
            index=False,
        )

    if not summaries:
        raise SystemExit("Nessun run valido.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    print()
    print("AlmenVision - valutazione ROI")
    print("=" * 118)
    print(summary_df.to_string(
        index=False,
        formatters={
            "precision": lambda x: f"{x:.3f}",
            "recall": lambda x: f"{x:.3f}",
            "f1": lambda x: f"{x:.3f}",
        }
    ))
    print()
    print("Le detection fuori ROI non contano come falsi positivi.")
    print("Le mele annotate devono appartenere solo alla zona operativa target.")


if __name__ == "__main__":
    main()
