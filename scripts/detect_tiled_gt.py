#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlmenVision - tiled inference SAHI sui frame della ground truth"
    )
    parser.add_argument("video", type=Path, help="Video sorgente")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/ground_truth.json"),
        help="Ground truth JSON",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Modello Ultralytics",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=640,
        help="Dimensione tile quadrata",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Overlap tra tile (0..1)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Device, es. "cpu" oppure "cuda:0"',
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/output/tiled_gt.csv"),
        help="CSV output compatibile con evaluate_runs.py",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/output/tiled_gt_frames"),
        help="Cartella frame annotati",
    )
    return parser.parse_args()


def load_gt_frames(path: Path) -> list[int]:
    if not path.exists():
        raise SystemExit(f"Ground truth non trovata: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    frames = sorted(int(k) for k in data.get("frames", {}).keys())

    if not frames:
        raise SystemExit("Nessun frame nella ground truth.")

    return frames


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video non trovato: {args.video}")

    if not 0.0 <= args.overlap < 1.0:
        raise SystemExit("--overlap deve essere >= 0 e < 1")

    gt_frames = load_gt_frames(args.ground_truth)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.images_dir.mkdir(parents=True, exist_ok=True)

    print("Caricamento modello SAHI/Ultralytics...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    total_predictions = 0
    start_all = time.perf_counter()

    with args.csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "timestamp_s",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_center",
                "y_center",
                "width",
                "height",
            ]
        )

        for n, frame_idx in enumerate(gt_frames, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok:
                print(f"[{n}/{len(gt_frames)}] frame {frame_idx}: LETTURA FALLITA")
                continue

            started = time.perf_counter()

            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=args.slice,
                slice_width=args.slice,
                overlap_height_ratio=args.overlap,
                overlap_width_ratio=args.overlap,
                progress_bar=False,
            )

            annotated = frame.copy()
            apple_predictions = []

            for pred in result.object_prediction_list:
                category_name = str(pred.category.name).lower()
                if category_name != "apple":
                    continue

                x1, y1, x2, y2 = pred.bbox.to_xyxy()
                confidence = float(pred.score.value)

                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))

                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2

                apple_predictions.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        x_center,
                        y_center,
                        width,
                        height,
                    )
                )

            for (
                x1,
                y1,
                x2,
                y2,
                confidence,
                x_center,
                y_center,
                width,
                height,
            ) in apple_predictions:
                writer.writerow(
                    [
                        frame_idx,
                        frame_idx / fps,
                        -1,
                        confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        x_center,
                        y_center,
                        width,
                        height,
                    ]
                )

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"APPLE {confidence:.0%}",
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            elapsed = time.perf_counter() - started
            total_predictions += len(apple_predictions)

            cv2.putText(
                annotated,
                (
                    f"AlmenVision TILED | {args.model} | "
                    f"slice {args.slice} | conf {args.conf:.2f}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_path = args.images_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), annotated)

            print(
                f"[{n}/{len(gt_frames)}] frame {frame_idx}: "
                f"{len(apple_predictions)} mele, {elapsed:.2f}s"
            )

    cap.release()

    elapsed_all = time.perf_counter() - start_all

    print()
    print("Tiled inference completata.")
    print(f"Frame valutati : {len(gt_frames)}")
    print(f"Detection mela : {total_predictions}")
    print(f"Tempo totale   : {elapsed_all:.1f}s")
    print(f"CSV            : {args.csv}")
    print(f"Frame annotati : {args.images_dir}")


if __name__ == "__main__":
    main()
