# Fourier cluster contour detection

This repository contains the paper's Fourier cluster contour detection network and one trained checkpoint. The default network combines point, temporal, and Fourier contour components.

## Contents

- `models/`: the released network configuration.
- `weights/best.pt`: the matching trained checkpoint (SHA-256: `64509098f44c8b62a63dee0fee26fd625e76f7fd3ad07e2a734007f489278587`).
- `detect_ft.py`: sequence inference.
- `val.py`: validation.
- `train.py`: training.
- `hyps/`: training hyperparameters.
- `data/infrared_e2.yaml`: an editable dataset path template.

## Setup

The release was checked with Python 3.9, PyTorch 2.1.2, and timm 1.0.30. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Inference

The model uses a five-frame temporal context. Point `--source` to a directory of frames ordered by filename:

```bash
python detect_ft.py --weights weights/best.pt --source path/to/image_sequence --device 0 --no-diagnostic --no-geometry-debug
```

Use `--recursive-sequences` when `--source` contains multiple sequence directories. See `python detect_ft.py --help` for output options.

## Validation and training

Edit `data/infrared_e2.yaml` so `path` points to your dataset root. The dataset images and labels are not included in this repository.

```bash
python val.py --data data/infrared_e2.yaml --weights weights/best.pt --device 0 --polygon
python train.py --data data/infrared_e2.yaml --device 0
```

The training command initializes the default backbone from timm and trains the released configuration. To fine-tune from the provided checkpoint, add `--weights weights/best.pt`.

## License

See `LICENSE` (GPL-3.0). The release retains the upstream YOLOv5 license terms.
