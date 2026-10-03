#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW = "AlmenVision - Add ROI to Ground Truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggiunge una ROI ai frame gia annotati in ground_truth.json "
            "e mantiene automaticamente solo le mele dentro la ROI."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/ground_truth/ground_truth.json"),
        help="Ground truth esistente con i punti mela",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ground_truth/ground_truth_roi.json"),
        help="Nuova ground truth con ROI e sole mele target",
    )
    return parser.parse_args()


def point_inside_polygon(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def draw_frame(
    image,
    roi_points: list[list[int]],
    all_apples: list[list[int]],
    target_apples: list[list[int]],
    frame_idx: int,
    pos: int,
    total: int,
):
    canvas = image.copy()

    # Tutte le mele originali: cerchi sottili
    for x, y in all_apples:
        cv2.circle(canvas, (int(x), int(y)), 7, (200, 200, 200), 1)

    # ROI
    if len(roi_points) >= 2:
        pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas,
            [pts],
            len(roi_points) >= 3,
            (0, 255, 255),
            3,
        )

        if len(roi_points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

    # Mele target dentro ROI
    for n, (x, y) in enumerate(target_apples, start=1):
        cv2.circle(canvas, (int(x), int(y)), 10, (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(n),
            (int(x) + 10, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    lines = [
        f"Frame {frame_idx} ({pos}/{total})",
        f"ROI punti: {len(roi_points)} | Mele originali: {len(all_apples)} | Target ROI: {len(target_apples)}",
        "Click SX = aggiungi punto ROI | U = undo | C = cancella ROI",
        "N/SPACE = prossimo | P = precedente | S = salva | Q/ESC = esci",
        "Grigio = annotazioni originali | Verde = mele mantenute nella ROI",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 27

    return canvas


def main() -> None:
    args = parse_args()

    if not args.ground_truth.exists():
        raise SystemExit(f"Ground truth non trovata: {args.ground_truth}")

    with args.ground_truth.open("r", encoding="utf-8") as fh:
        original = json.load(fh)

    frames_data = original.get("frames", {})
    if not frames_data:
        raise SystemExit("La ground truth non contiene frame annotati.")

    frame_keys = sorted(frames_data.keys(), key=int)

    # Carica immagini associate ai frame
    images = {}
    for key in frame_keys:
        info = frames_data[key]
        image_path = Path(info["image"])
        if not image_path.exists():
            raise SystemExit(
                f"Immagine del frame {key} non trovata: {image_path}"
            )
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(
                f"Impossibile leggere immagine del frame {key}: {image_path}"
            )
        images[key] = image

    # Se output esiste, riprende eventuale lavoro già fatto
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as fh:
            output = json.load(fh)
    else:
        output = {
            "video": original.get("video"),
            "fps": original.get("fps"),
            "total_frames": original.get("total_frames"),
            "frames": {},
        }

    state = {
        "pos": 0,
        "roi": [],
    }

    def current_key() -> str:
        return frame_keys[state["pos"]]

    def load_current() -> None:
        key = current_key()
        info = output["frames"].get(key, {})
        state["roi"] = [list(map(int, p)) for p in info.get("roi", [])]

    def compute_target_apples() -> list[list[int]]:
        key = current_key()
        apples = frames_data[key].get("apples", [])
        if len(state["roi"]) < 3:
            return []
        return [
            [int(x), int(y)]
            for x, y in apples
            if point_inside_polygon((float(x), float(y)), state["roi"])
        ]

    def persist_current() -> None:
        key = current_key()
        source = frames_data[key]
        target_apples = compute_target_apples()

        output["frames"][key] = {
            "timestamp_s": source.get("timestamp_s"),
            "image": source.get("image"),
            "roi": state["roi"],
            "apples": target_apples,
            "apple_count": len(target_apples),
            "original_apple_count": len(source.get("apples", [])),
        }

    def save_output() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["roi"].append([x, y])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    load_current()

    while True:
        key = current_key()
        source = frames_data[key]
        all_apples = [
            [int(x), int(y)]
            for x, y in source.get("apples", [])
        ]
        target_apples = compute_target_apples()

        display = draw_frame(
            images[key],
            state["roi"],
            all_apples,
            target_apples,
            int(key),
            state["pos"] + 1,
            len(frame_keys),
        )
        cv2.imshow(WINDOW, display)
        pressed = cv2.waitKey(30) & 0xFF

        if pressed in (ord("q"), 27):
            persist_current()
            save_output()
            break

        elif pressed == ord("u"):
            if state["roi"]:
                state["roi"].pop()

        elif pressed == ord("c"):
            state["roi"].clear()

        elif pressed == ord("s"):
            persist_current()
            save_output()
            print(f"Salvato: {args.output}")

        elif pressed in (ord("n"), 32):
            persist_current()
            save_output()
            if state["pos"] < len(frame_keys) - 1:
                state["pos"] += 1
                load_current()

        elif pressed == ord("p"):
            persist_current()
            save_output()
            if state["pos"] > 0:
                state["pos"] -= 1
                load_current()

    cv2.destroyAllWindows()

    total_original = sum(
        len(info.get("apples", []))
        for info in frames_data.values()
    )
    total_target = sum(
        info.get("apple_count", 0)
        for info in output["frames"].values()
    )

    print()
    print("ROI aggiunta alla ground truth.")
    print(f"Output: {args.output}")
    print(f"Mele annotate originali: {total_original}")
    print(f"Mele target dentro ROI:  {total_target}")


if __name__ == "__main__":
    main()
