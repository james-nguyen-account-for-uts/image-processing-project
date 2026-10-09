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

The default configuration uses GPU `0` when it is available. If CUDA is not
installed, set `device: auto` in `configs/default.yaml` to use CPU fallback.

(for Windows) If PowerShell blocks activation, run the following command once in that terminal and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The `pip install -e .` command imports the local `fall_detection` package and creates the three terminal commands used below. Run the installation commands again whenever `requirements.txt` changes.

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

Activate `.venv` whenever you open a new terminal:

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

This implements the report's two rules on annotated **person regions**:

- Aspect ratio: predict Fall when `r > aspect_threshold`, where
  `r = (normalised_box_width * image_width) / (normalised_box_height * image_height)`.
- Aspect ratio plus position: also require `cy >= position_threshold`, where
  `cy` is the YOLO-normalised vertical box centre (0 at the top, 1 at the bottom).

The baseline receives boxes from the dataset annotations and does not locate
people in arbitrary images. `fall` is the positive class; `up` is the negative
class. `0 person` annotations and empty-label images are recorded separately
and excluded from person-region scoring, rather than counted as true negatives.
Missing label files and malformed annotations cause an error.

After installing the project again with `pip install -e .`, run:

```bash
fall-geometry --config configs/geometry.yaml
```

Or run the module directly from the repository root:

```bash
python -m fall_detection.geometry --config configs/geometry.yaml
```

Configuration paths are relative to the working directory. Dataset split paths
are relative to the dataset YAML directory, or its optional `path` entry; this
baseline accepts `.../images` directories with sibling `labels` directories.
Image headers provide the original dimensions; images are not resized for this
baseline. This matters because a ratio of normalised width to normalised height
alone is incorrect for non-square images.

Both rules are selected using **validation only**, maximising Fall F1 and then
Fall recall. Threshold candidates cover all distinct validation decisions.
Further ties favour lower position thresholds and then lower aspect thresholds;
if both variants have the same validation F1 and recall, the aspect-only rule is
recommended. Thresholds and the recommended variant are saved before test data
is opened. The training split is not used because this rule has no learned
weights. Test results never select thresholds or the recommended variant.

The default output directory is `runs/geometry/`:

- `model.json`: frozen thresholds, class mapping and selection procedure.
- `metrics.json`: split counts, Fall precision/recall/F1, TP/FP/TN/FN, Up-region
  false-positive rate, confusion matrices, timing, environment and input hashes.
- `summary.md`: validation and test comparison.
- `{split}_{rule}_predictions.csv`: one record per annotated person region,
  including source image, annotation line number, features and true/predicted labels.
- `{split}_{rule}_errors.csv`: false positives and false negatives.
- `{split}_{rule}_confusion.csv`: a labelled binary confusion matrix.
- `{split}_excluded.csv`: excluded no-person annotations and empty-label images.

To reuse saved thresholds without selecting them again:

```bash
fall-geometry --config configs/geometry.yaml --model runs/geometry/model.json
```

Timing uses one warmup pass and 25 measured passes by default. It includes
geometry calculation and rule prediction, and excludes disk I/O, person
localisation, threshold search and output writing. It is a person-region timing,
not end-to-end image detection latency. The input fingerprints cover image
identifiers, dimensions and annotation bytes; they do not hash image pixels or
check cross-split duplicates.

The completed local experiment selected `r > 0.5128939828080229`. Validation F1
was 0.9477 and test F1 was 0.9202. The best position threshold was 0, so the
position condition provided no additional benefit under this tested rule family.
See [the experiment note](docs/geometry_experiment.md) for counts and error analysis.

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
