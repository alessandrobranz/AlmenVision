#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valuta uno o più run AlmenVision confrontando le detection CSV "
            "con la ground truth annotata manualmente."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/ground_truth.json"),
        help="Ground truth JSON prodotto da ground_truth.py",
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Uno o più CSV prodotti da detect_video.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Cartella output dei report",
    )
    return parser.parse_args()


def load_ground_truth(path: Path) -> dict[int, list[tuple[float, float]]]:
    if not path.exists():
        raise SystemExit(f"Ground truth non trovata: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    frames = raw.get("frames", {})
    gt: dict[int, list[tuple[float, float]]] = {}

    for frame_key, frame_info in frames.items():
        apples = frame_info.get("apples", [])
        gt[int(frame_key)] = [(float(x), float(y)) for x, y in apples]

    if not gt:
        raise SystemExit("La ground truth non contiene frame annotati.")

    return gt


def point_inside_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def squared_distance(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def match_frame(
    gt_points: list[tuple[float, float]],
    detections: pd.DataFrame,
) -> tuple[int, int, int]:
    """
    Match one-to-one:
    - ogni detection può corrispondere a una sola mela GT
    - ogni mela GT può essere usata una sola volta
    - una detection è corretta se contiene il punto annotato della mela
    - se contiene più punti, sceglie quello più vicino al centro della box
    - le detection vengono valutate in ordine di confidence decrescente
    """
    if detections.empty:
        return 0, 0, len(gt_points)

    detections = detections.sort_values("confidence", ascending=False)
    unmatched_gt = set(range(len(gt_points)))

    tp = 0
    fp = 0

    for _, row in detections.iterrows():
        box = (
            float(row["x1"]),
            float(row["y1"]),
            float(row["x2"]),
            float(row["y2"]),
        )
        center = box_center(box)

        candidates = [
            gt_idx
            for gt_idx in unmatched_gt
            if point_inside_box(gt_points[gt_idx], box)
        ]

        if not candidates:
            fp += 1
            continue

        best_gt = min(
            candidates,
            key=lambda idx: squared_distance(gt_points[idx], center),
        )

        unmatched_gt.remove(best_gt)
        tp += 1

    fn = len(unmatched_gt)
    return tp, fp, fn


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_run(
    csv_path: Path,
    gt: dict[int, list[tuple[float, float]]],
) -> tuple[dict, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required_columns = {
        "frame",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path}: colonne mancanti: {', '.join(sorted(missing))}"
        )

    frame_rows = []

    for frame_idx in sorted(gt):
        gt_points = gt[frame_idx]
        detections = df[df["frame"] == frame_idx]

        tp, fp, fn = match_frame(gt_points, detections)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        frame_rows.append(
            {
                "frame": frame_idx,
                "gt_apples": len(gt_points),
                "detections": len(detections),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    frame_df = pd.DataFrame(frame_rows)

    tp_total = int(frame_df["tp"].sum())
    fp_total = int(frame_df["fp"].sum())
    fn_total = int(frame_df["fn"].sum())
    gt_total = int(frame_df["gt_apples"].sum())
    detections_total = int(frame_df["detections"].sum())

    precision = safe_div(tp_total, tp_total + fp_total)
    recall = safe_div(tp_total, tp_total + fn_total)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "run": csv_path.stem,
        "frames_evaluated": len(frame_df),
        "gt_apples": gt_total,
        "detections": detections_total,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return summary, frame_df


def main() -> None:
    args = parse_args()

    gt = load_ground_truth(args.ground_truth)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for csv_path in args.csv_files:
        try:
            summary, frame_df = evaluate_run(csv_path, gt)
        except FileNotFoundError:
            print(f"ATTENZIONE: file non trovato: {csv_path}")
            continue

        summaries.append(summary)

        per_frame_path = args.output_dir / f"{csv_path.stem}_per_frame.csv"
        frame_df.to_csv(per_frame_path, index=False)

    if not summaries:
        raise SystemExit("Nessun run valido da valutare.")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[
        [
            "run",
            "frames_evaluated",
            "gt_apples",
            "detections",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ]
    ]

    summary_path = args.output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("AlmenVision - valutazione contro ground truth")
    print("=" * 118)
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "precision": lambda x: f"{x:.3f}",
                "recall": lambda x: f"{x:.3f}",
                "f1": lambda x: f"{x:.3f}",
            },
        )
    )

    print()
    print("Interpretazione:")
    print("- TP = mele reali correttamente rilevate")
    print("- FP = detection che non corrispondono a una mela annotata")
    print("- FN = mele reali non rilevate")
    print("- Precision = TP / (TP + FP)")
    print("- Recall    = TP / (TP + FN)")
    print("- F1        = media armonica tra precision e recall")
    print()
    print(f"Report riepilogativo: {summary_path}")
    print(f"Report per-frame:      {args.output_dir}/<run>_per_frame.csv")


if __name__ == "__main__":
    main()
