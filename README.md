# Fall detection: three methods

Run `python main.py` from the project virtual environment. The menu offers:

1. **Bounding box** — width/height ratio, optionally combined with vertical position.
2. **HOG–SVM** — grayscale HOG features from person crops, classified with a linear SVM.
3. **YOLOv8** — YOLOv8n fine-tuned on the dataset's bounding-box annotations.
0. **Exit** — typing `exit` also works.

Selecting a method opens **Train / calibrate**, **Evaluate on test set**, and **Detect image, video, or webcam**. Train the selected method before evaluation or detection. Missing models and invalid inputs return an explanatory message and leave the menu usable. Camera/video windows close with Q or Esc; press a key to dismiss a still image.

## Setup and launch

Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python main.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

The launcher resolves paths from its own directory, so you can also invoke it by absolute path from elsewhere. Direct CLI commands below should be run from the repository root. The first use of YOLOv8 downloads pretrained weights and needs internet access. Training YOLOv8 defaults to 30 epochs; edit `configs/default.yaml` for epochs, image size, batch size, and device (`auto`, `cpu`, `mps` for supported Macs, or `0` for CUDA).

## Structure

```text
main.py                    # Interactive launcher
src/
  bounding_box/            # Geometry calibration, evaluation and detection
  hog_svm/                 # HOG extraction, SVM fitting and inference
  yolov8/                  # YOLOv8 training, evaluation and inference
  common/                  # Annotation reader, metrics and person localisation
configs/
  geometry.yaml
  hog_svm.yaml
  default.yaml             # YOLOv8 defaults
  gpu.yaml                 # YOLOv8 CUDA settings
data/
  data.yaml
  train/{images,labels}/
  valid/{images,labels}/
  test/{images,labels}/
runs/                      # Ignored generated models, predictions and metrics
```

The old `src/fall_detection` package has been replaced. The legacy `fall-train`, `fall-evaluate`, `fall-detect`, and `fall-geometry` commands remain available after reinstalling with `pip install -e .`; the per-method commands below are preferred.

## Dataset

All methods use `data/data.yaml`, with image paths `train/images`, `valid/images`, and `test/images` relative to that file. The supplied Roboflow Fall v1 export has classes `0 person`, `fall`, and `up`. Keep dataset files and weights out of Git.

Bounding-box calibration selects shape/position thresholds using **validation only**. HOG–SVM fits a `StandardScaler` and balanced linear SVM on **training only**, selects C using validation fall F1 (then recall), and saves the selected fitted pipeline. Both evaluate frozen models on the test set. YOLOv8 trains on training images and uses validation during training; the menu evaluates on test.

## Commands

```bash
python -m bounding_box train
python -m bounding_box evaluate
python -m bounding_box detect --source 0

python -m hog_svm train
python -m hog_svm evaluate
python -m hog_svm detect --source 0

python -m yolov8 train
python -m yolov8 evaluate
python -m yolov8 detect --source 0
```

Each supports `--config`. Detection accepts image/video paths or a camera index and `--no-show`. Bounding box and HOG–SVM also accept `--labels path/to/labels.txt` for a still image, using the supplied person boxes instead of automatic localisation. Annotation class values identify person regions but do not determine predicted Fall/No Fall labels.

Default models:

- Bounding box: `runs/geometry/model.json`
- HOG–SVM: `runs/hog_svm/model.joblib`
- YOLOv8: `runs/yolov8/train/weights/best.pt`

Retraining replaces the method's default model. Only load trusted joblib model files. HOG and geometry evaluation save metrics and individual predictions; their detection saves JSONL results and annotated still images. YOLOv8 saves its standard annotated images/videos and evaluation output.

## What the methods measure

Geometry and HOG–SVM classify **individual person regions**. Their dataset evaluation uses ground-truth boxes and excludes empty/no-person regions; it measures posture classification, not person localisation. At inference, their automatic image/video/camera mode uses a **pretrained YOLOv8 person detector**, followed by the selected geometry or HOG classifier. This is explicitly a hybrid localisation pipeline: YOLO does not supply their fall decision. You can supply annotation boxes to run these classifiers without YOLO.

The third method learns boxes and classes directly from dataset images. Its mAP scores cannot be compared directly with person-region F1. The export contains only one `0 person` training annotation and none in validation/test, so it does not establish no-person rejection performance. These static posture methods do not track a fall event over time or distinguish every normal lying-down activity from a fall.

## Tests

```bash
python -m pytest -q
```

Tests cover geometry rules, annotation validation, frozen-model evaluation, camera cleanup, menu routing, HOG fitting/selection splits, saved classifier inference, and headless image detection. Actual webcam permissions and camera framing need to be checked on the machine used for the demonstration.

## Evaluation charts in one window

Choose **4 — Display evaluation charts** from the main menu. The window shows a column for each method, with score bars above its confusion matrix. Bounding box uses its validation-selected rule. YOLOv8 shows fall-class precision/recall/F1 plus overall mAP50 and mAP50–95. Matrix rows are true labels and columns are predictions for all methods.

This option only reads saved **test** evaluations; it does not start training or inference. Missing results appear as pending. Evaluate each method first (method number → option 2), then reopen the chart. If the launcher was already open before this update, finish training, exit it, and restart `python main.py` to load the new menu and evaluation exporter. Existing YOLO evaluations without the new JSON report need to be evaluated once again.

The same chart is saved to `runs/comparison/test_evaluations.png` for presentations. To export without opening a window, run `python -m common.comparison --no-show`. Custom experiments can be viewed by updating the relevant default method config's output directory. The display is a snapshot; close and reopen it after evaluating again.
