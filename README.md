# Fall Detection Foundation

A small Python foundation for detecting whether a person is in a fall posture. It includes YOLO training and inference, plus a bounding-box geometry baseline using supplied person annotations. The repository separates dataset setup, training, evaluation, and inference so each part can be developed independently.

## Dataset

Use **Version 1** of the Roboflow Universe dataset:

- [Fall dataset — Roboflow Universe](https://universe.roboflow.com/search-kinvz/fall-ljaaq/dataset/1)

Download/export it in **YOLO format** and extract it into `data/fall/`. The exported dataset should provide a `data.yaml` file and train/validation/test folders. Dataset files are intentionally excluded from Git because image datasets are usually too large for a normal repository.

Before training, open `data/fall/data.yaml` and confirm that:

- its image paths are correct for your machine;
- its class names match the labels in the exported dataset;
- the training and validation splits both contain labelled examples.

If this dataset contains only a `fall` class, the model will learn to detect visible falls, while images with no labelled fall act as background. For a stronger fall-versus-no-fall classifier later, add representative non-fall activities such as sitting, lying normally, bending, and walking.

## Repository structure

```text
fall-detection-foundation/
├── configs/default.yaml       # Shared experiment settings
├── configs/geometry.yaml      # Geometry experiment settings
├── data/README.md             # Dataset placement guide
├── src/fall_detection/
│   ├── config.py              # Loads and validates configuration
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation entry point
│   ├── geometry.py            # Validation-selected geometry baselines
│   └── detect.py              # Image/video/camera inference
├── tests/test_config.py       # Starter unit tests
├── .gitignore
├── pyproject.toml
└── README.md
```

## Environment setup

Python 3.10 or newer is recommended. Each teammate should create a local `.venv` so the project packages do not interfere with other Python projects.

### macOS or Linux

From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

From the repository root, run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For NVIDIA GPU training on Windows, replace the CPU PyTorch packages installed
by the command above with a CUDA build, then confirm that CUDA is available:

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The default configuration uses `device: auto`. Use `configs/gpu.yaml` to explicitly select GPU `0` when CUDA is installed.

(for Windows) If PowerShell blocks activation, run the following command once in that terminal and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The `pip install -e .` command imports the local `fall_detection` package and creates the terminal commands used below. Run the installation commands again whenever `requirements.txt` changes.

On Windows, register the four project commands in your user PATH to run them from
ordinary PowerShell or Command Prompt without activating `.venv`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_local_commands.ps1
```

Close and reopen the terminal, then run `fall-train`, `fall-evaluate`, `fall-detect`
or `fall-geometry` from the repository root. These entries use this checkout's
`.venv`; keep the project at its current location or rerun registration after moving it.

To leave the virtual environment later, run:

```bash
deactivate
```

## Dataset setup

1. Open [Version 1 of the Fall dataset on Roboflow Universe](https://universe.roboflow.com/search-kinvz/fall-ljaaq/dataset/1).
2. Download or export the dataset in YOLO format.
3. Extract it into `data/fall/`.
4. Confirm that `data/fall/data.yaml` exists and that its paths and class names are correct.

The final structure should resemble:

```text
data/fall/
├── data.yaml
├── train/
├── valid/
└── test/        # Optional if the dataset has no test split
```

## Running the project

Activate `.venv` whenever you open a new terminal, unless you registered the Windows command entries above:

```bash
source .venv/bin/activate        # macOS/Linux
```

```powershell
.venv\Scripts\Activate.ps1      # Windows PowerShell
```

Then train the model:

```bash
fall-train --config configs/default.yaml
```

Evaluate the best trained weights:

```bash
fall-evaluate --config configs/default.yaml
```

Run live detection using the default camera:

```bash
fall-detect --config configs/default.yaml --source 0
```

For `fall-detect`, `--source` may also be an image path, video path, stream URL, or a different camera index. For example:

```bash
fall-detect --config configs/default.yaml --source examples/test_video.mp4
```

Trained weights are expected at `runs/fall_detection/train/weights/best.pt` by default.

Run the tests with:

```bash
pytest
```

## Bounding-box geometry baseline

Use `--method geometry` with the same three commands. Training selects thresholds on validation;
evaluation uses saved rules (test split by default). Image detection requires supplied person boxes;
camera detection locates people automatically before applying Geometry rules.

```bash
fall-train --method geometry
fall-evaluate --method geometry
fall-detect --method geometry --source photo.jpg --bbox 100 100 500 300
fall-detect --method geometry --source 0
```

Repeat `--bbox` for multiple people, or use `--labels photo.txt` for YOLO annotations.
Camera mode uses `yolov8n.pt` for person boxes only; Geometry decides Fall/No Fall. Press `Q`/`Esc`
or close the window to stop. Use `--detector-weights other.pt` to change the person detector
(classes must include `person` or the configured Fall/Up classes), and `--detector-confidence 0.4`
to change its confidence threshold. Frame results are saved as JSONL; use `--output camera.jsonl`
for a custom path or `--no-show` for headless operation (stop with Ctrl+C).
Rules and results are saved to `runs/geometry/`; evaluation and detection accept `--weights` for another `model.json`.
YOLO remains the default method. The original `fall-geometry` experiment command also remains available.

Without installed console commands, use `python scripts/run_fall.py train --method geometry`
(replace `train` with `evaluate` or `detect` and pass the same options).

## Simple team plan

The work is deliberately split into small pieces. Nobody needs to redesign the repository.

### Teammate 1 — dataset and training

1. Download the dataset and confirm `data/fall/data.yaml` works.
2. Review class balance and inspect a small sample of labels.
3. Adjust the basic training settings in `configs/default.yaml` if needed.
4. Run `fall-train` and record the best model path and training metrics.

Expected output: a trained `best.pt` model and a short note containing dataset counts, settings, and results.

### Teammate 2 — inference and testing

1. Test `fall-detect` on several images and one short video.
2. Check the confidence threshold and note obvious false positives/negatives.
3. Add one or two tests or input-validation checks.
4. Write brief reproduction steps for anything that fails.

Expected output: working inference examples and a short test summary.

### Project owner / integration

1. Review pull requests and keep configuration values consistent.
2. Run training, evaluation, and inference end to end.
3. Compare results and decide what needs improvement.
4. Take over either teammate section if it is unfinished; the modules are independent and already have runnable entry points.

## Working agreement

- Create a short branch for each task, for example `dataset-training` or `inference-testing`.
- Do not commit datasets, model weights, or generated `runs/` files.
- Keep paths and thresholds in `configs/default.yaml` instead of hard-coding them.
- Make small pull requests and include the exact command used for testing.

## Current scope

This foundation provides a straightforward prototype, not a clinical or emergency-response system. Public-dataset results may not generalise to different people, rooms, camera angles, lighting, occlusion, or fall-like daily activities.

## Image-level evaluation (Chovy)

See [LOCAL_EVALUATION.md](LOCAL_EVALUATION.md) for validation-only threshold selection, frozen test evaluation, prediction CSVs, and error analysis. Detection mAP and geometry region-level evaluation remain separate.
