# Fourier cluster contour detection

This repository contains the paper's Fourier cluster contour detection network and one trained checkpoint. The default network combines point, temporal, and Fourier contour components.

## Demo videos

The three sequences show point detections and predicted Fourier contours over 100 frames at 3 fps. Green boxes mark matched detections, red boxes mark missed targets, and yellow boxes mark false positives. Cyan curves are predicted contours.

### Cloud 5

https://github.com/user-attachments/assets/653435b2-f18c-4643-9949-49ffb923a12a

### Ground 3

https://github.com/user-attachments/assets/e10ad813-745c-48a5-b28d-4185e3f50fae

### Ground 6

https://github.com/user-attachments/assets/cefc1e89-6275-45a9-ab70-ea5864e353b3

## Contents

- `models/`: the released network configuration.
- `weights/best.pt`: the matching trained checkpoint (SHA-256: `64509098f44c8b62a63dee0fee26fd625e76f7fd3ad07e2a734007f489278587`).
- `detect_ft.py`: sequence inference.
- `val.py`: validation.
- `train.py`: training.
- `hyps/`: training hyperparameters.
- `data/infrared_e2.yaml`: an editable dataset path template.

## Dataset

Download [data_mix.zip](https://drive.google.com/file/d/1VwQMB_SzWLIUzNBbWHDssBtxvUwsZORo/view?usp=drive_link) from the [FSTR-Net dataset folder on Google Drive](https://drive.google.com/drive/folders/1svKWSU2gHHHCPFCSZJvMaRQdQ8LQbb3V). The archive contains 12,550 files (1,829,687,501 bytes; SHA-256: `1c79c34f7fb87554b903035faa5352a5e04ab8d4ac59dcb3c814ff2d9b612a39`). Extract it and set `path` in `data/infrared_e2.yaml` to the extracted `data_mix` directory.

The validation split also includes all 80 frames of `test` and the first 50 frames of `test_new` from `Experiment_eva`, with their matching labels. The release archive omits one exact duplicate validation image and backup `.bak` files. The original dataset and sequence folders were not modified.

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
