#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


WINDOW = "AlmenVision Ground Truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotazione manuale di mele su frame campione per AlmenVision."
    )
    parser.add_argument("video", type=Path, help="Video sorgente")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Numero di frame da campionare (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ground_truth/ground_truth.json"),
        help="File JSON di output",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("data/ground_truth/frames"),
        help="Cartella in cui salvare i frame campione",
    )
    return parser.parse_args()


def choose_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Numero di frame non valido.")
    count = max(1, min(count, total_frames))

    if count == 1:
        return [total_frames // 2]

    # Evita primo e ultimo frame, spesso poco rappresentativi.
    start = max(0, int(total_frames * 0.05))
    end = min(total_frames - 1, int(total_frames * 0.95))

    if count == 1 or end <= start:
        return [total_frames // 2]

    step = (end - start) / (count - 1)
    return [round(start + i * step) for i in range(count)]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {"video": None, "frames": {}}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_data(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def draw_overlay(image, points: list[list[int]], frame_idx: int, current: int, total: int):
    canvas = image.copy()

    for n, (x, y) in enumerate(points, start=1):
        cv2.circle(canvas, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(canvas, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            str(n),
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    overlay_lines = [
        f"Frame {frame_idx}  ({current}/{total})",
        f"Mele annotate: {len(points)}",
        "Click SX = mela | U = undo | C = cancella frame",
        "N/SPACE = prossimo | P = precedente | S = salva | Q/ESC = esci",
    ]

    y0 = 32
    for i, line in enumerate(overlay_lines):
        y = y0 + i * 28
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video non trovato: {args.video}")

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Impossibile aprire il video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_indices = choose_frame_indices(total_frames, args.count)

    existing = load_existing(args.output)
    data = {
        "video": str(args.video),
        "fps": fps,
        "total_frames": total_frames,
        "frames": existing.get("frames", {}),
    }

    frames: dict[int, any] = {}

    print("Estraggo i frame campione...")
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"ATTENZIONE: impossibile leggere frame {frame_idx}")
            continue

        frames[frame_idx] = frame
        out_path = args.frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)

        key = str(frame_idx)
        if key not in data["frames"]:
            data["frames"][key] = {
                "timestamp_s": frame_idx / fps,
                "image": str(out_path),
                "apples": [],
            }

    cap.release()

    valid_indices = [idx for idx in frame_indices if idx in frames]
    if not valid_indices:
        raise SystemExit("Nessun frame valido estratto.")

    state = {
        "pos": 0,
        "points": [],
        "dirty": False,
    }

    def load_current_points() -> None:
        idx = valid_indices[state["pos"]]
        state["points"] = [
            [int(p[0]), int(p[1])]
            for p in data["frames"][str(idx)].get("apples", [])
        ]
        state["dirty"] = False

    def persist_current() -> None:
        idx = valid_indices[state["pos"]]
        data["frames"][str(idx)]["apples"] = state["points"]
        data["frames"][str(idx)]["apple_count"] = len(state["points"])
        state["dirty"] = False

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"].append([x, y])
            state["dirty"] = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_callback)

    load_current_points()

    while True:
        idx = valid_indices[state["pos"]]
        frame = frames[idx]

        display = draw_overlay(
            frame,
            state["points"],
            idx,
            state["pos"] + 1,
            len(valid_indices),
        )

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # Q o ESC
            persist_current()
            save_data(args.output, data)
            break

        if key == ord("u"):
            if state["points"]:
                state["points"].pop()
                state["dirty"] = True

        elif key == ord("c"):
            state["points"].clear()
            state["dirty"] = True

        elif key == ord("s"):
            persist_current()
            save_data(args.output, data)
            print(f"Salvato: {args.output}")

        elif key in (ord("n"), 32):  # N o SPACE
            persist_current()
            save_data(args.output, data)
            if state["pos"] < len(valid_indices) - 1:
                state["pos"] += 1
                load_current_points()

        elif key == ord("p"):
            persist_current()
            save_data(args.output, data)
            if state["pos"] > 0:
                state["pos"] -= 1
                load_current_points()

    cv2.destroyAllWindows()

    total_gt = sum(
        len(frame_info.get("apples", []))
        for frame_info in data["frames"].values()
        if str(frame_info.get("image", "")).startswith(str(args.frames_dir))
    )

    print()
    print("Ground truth salvata.")
    print(f"File: {args.output}")
    print(f"Frame annotati: {len(valid_indices)}")
    print(f"Mele annotate complessive: {total_gt}")


if __name__ == "__main__":
    main()
