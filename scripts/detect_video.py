#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlmenVision V0.1 - apple detection/tracking on prerecorded video"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/almenvision_output.mp4"),
        help="Annotated output video",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/output/detections.csv"),
        help="CSV output",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name/path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview while processing",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Run YOLO detection only, without ByteTrack",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.input}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Cannot create output video: {args.output}")

    # COCO class 47 = apple
    apple_class_id = 47

    with args.csv.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
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

        frame_idx = 0
        start = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.no_track:
                result = model.predict(
                    frame,
                    classes=[apple_class_id],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]
            else:
                result = model.track(
                    frame,
                    persist=True,
                    classes=[apple_class_id],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]

            annotated = frame.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    x1, y1, x2, y2 = map(
                        int,
                        result.boxes.xyxy[i].cpu().tolist(),
                    )

                    confidence = float(result.boxes.conf[i].cpu())

                    track_id = -1
                    if (
                        not args.no_track
                        and result.boxes.id is not None
                    ):
                        track_id = int(result.boxes.id[i].cpu())

                    box_width = x2 - x1
                    box_height = y2 - y1
                    x_center = x1 + box_width / 2
                    y_center = y1 + box_height / 2

                    if args.no_track:
                        label = f"APPLE {confidence:.0%}"
                    else:
                        label = f"APPLE #{track_id} {confidence:.0%}"

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    csv_writer.writerow(
                        [
                            frame_idx,
                            frame_idx / fps_in,
                            track_id,
                            confidence,
                            x1,
                            y1,
                            x2,
                            y2,
                            x_center,
                            y_center,
                            box_width,
                            box_height,
                        ]
                    )

            elapsed = max(time.perf_counter() - start, 1e-9)
            proc_fps = (frame_idx + 1) / elapsed

            mode = "DETECT" if args.no_track else "TRACK"

            cv2.putText(
                annotated,
                (
                    f"AlmenVision V0.1 | {mode} | "
                    f"{args.model} | imgsz {args.imgsz} | "
                    f"conf {args.conf:.2f} | FPS {proc_fps:.1f}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            if args.show:
                cv2.imshow("AlmenVision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed_total = max(time.perf_counter() - start, 1e-9)
    avg_fps = frame_idx / elapsed_total if frame_idx else 0.0

    print()
    print("AlmenVision processing complete.")
    print(f"Mode   : {'detection' if args.no_track else 'tracking'}")
    print(f"Model  : {args.model}")
    print(f"imgsz  : {args.imgsz}")
    print(f"conf   : {args.conf}")
    print(f"Frames : {frame_idx}")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"Video  : {args.output}")
    print(f"CSV    : {args.csv}")


if __name__ == "__main__":
    main()
