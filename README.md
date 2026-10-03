# AlmenVision

Computer vision per Apple Harvester.

## Attiva ambiente

```bash
cd /home/branza/workspace/AlmenVision
source .venv/bin/activate
```

## Test installazione

```bash
pytest -q
ruff check .
```

## Test webcam

```bash
python scripts/test_webcam.py
```

## Analisi video

Copia il video in `data/input/` e lancia:

```bash
python scripts/detect_video.py data/input/video.mp4 --show
```

Output:
- `data/output/almenvision_output.mp4`
- `data/output/detections.csv`

Il modello YOLO viene scaricato automaticamente al primo utilizzo.
