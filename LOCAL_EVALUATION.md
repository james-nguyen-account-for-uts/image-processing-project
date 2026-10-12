# Local image-level evaluation

## Status

The implementation is ready for local testing. No real dataset or trained weights
were supplied, so no real performance results or error-cause claims are included.
Unit/integration tests use explicitly synthetic images and a mocked detector.

The project defaults now select **yolov8n.pt**, matching the agreed report.
This changes future model selection, not existing trained checkpoints. Do not
rename a YOLO11 checkpoint and describe it as YOLOv8. Geometry camera localisation
also defaults to YOLOv8n; it remains separate from the fall classifier.

## Prerequisites

Run from this project root. Install requirements in a virtual environment.
The test environment prepared here contains only the lightweight testing dependencies;
real inference additionally needs the project's Ultralytics/PyTorch dependencies.

1. Obtain the agreed dataset `data.yaml`, with explicit `val` and `test` image directories.
   Relative paths resolve against the YAML's parent and optional `path` field.
   Correct exported paths if they point outside the actual dataset.
2. Each image needs a corresponding `labels/*.txt`; empty files are accepted,
   missing files are errors. Nested image directories are supported; list files
   and multiple-directory split specifications are intentionally rejected.
3. Obtain a trained fall detector `best.pt`. Generic COCO pretrained weights are
   not suitable for evaluation because they do not carry the fall label mapping.
4. Review the meaning of `0 person` and label completeness. The evaluator preserves
   `no_person_label` and `empty_annotation` as separate annotation groups; neither
   is automatically described as a verified empty scene.

## Stage 1: validation-only calibration

```powershell
.venv\Scripts\python.exe scripts/run_image_evaluation.py calibrate --config configs/default.yaml --weights runs/fall_detection/train/weights/best.pt --output runs/image_eval/validation_01
```

The command reads **only val**, predicts at confidence floor 0.001, and searches
distinct attainable thresholds at/above that floor. Fall means highest retained
fall confidence **strictly exceeds** the threshold. It maximises Fall F1, then
Recall; remaining ties choose the lowest threshold. Both true classes must exist
in validation. The detector uses fixed class-aware NMS (IoU 0.7, max_det 300),
square letterboxing, no augmentation and batch size one.

`calibration.json` freezes the threshold, inference settings, class mapping,
checkpoint SHA-256 and validation image-content hashes. `threshold_search.csv`
provides the selection evidence. Do not calibrate on test data.

## Stage 2: frozen test evaluation

```powershell
.venv\Scripts\python.exe scripts/run_image_evaluation.py test --config configs/default.yaml --weights runs/fall_detection/train/weights/best.pt --calibration runs/image_eval/validation_01/calibration.json --output runs/image_eval/test_01
```

This reads **only test**, reuses the frozen threshold and inference settings, and
rejects changed weights/classes/input size or exact validation/test image overlap.
The checkpoint is compared by content, not filename. Existing output directories
are rejected to avoid overwriting experiment evidence.

Each stage exports:

- `predictions.csv`: source image, truth, prediction, fall score, TP/FP/TN/FN,
  negative annotation group, processing time and content/label hashes.
- `errors.csv`: only FP and FN cases.
- `metrics.json`: counts, Precision/Recall/F1, subgroup FPRs with denominators,
  confusion matrix, configuration, environment and provenance.
- `summary.md`: concise image-level results and limitations.
- `error_review.md`: worksheet for manual inspection; no fabricated causes.

No detection means score zero. A missed fall is FN. Any annotated fall makes the
whole image positive, including mixed fall/upright scenes. Zero-denominator
Precision/Recall/F1 are recorded as zero. Absent negative groups are omitted.
Timing includes image loading through CPU confidence extraction after one warm-up;
it excludes model loading and data auditing. Treat it as measured local processing
time, not camera fall-event response latency.

## Detection mAP remains separate

```powershell
.venv\Scripts\python.exe scripts/run_fall.py evaluate --method yolo --split test --weights runs/fall_detection/train/weights/best.pt
```

This existing command reports detection mAP. It does not replace the image-level
workflow. Geometry's annotated-person-region scores must be reported separately.

## Remaining real-experiment work

- Verify dataset semantics and train/validation/test source separation. Exact
  validation/test hashing here does not check near-duplicates, train overlap or
  frames from the same recording.
- Record actual trained-model identity and training settings; obtain group members'
  geometry and HOG-SVM outputs. HOG-SVM is outside this evaluation implementation.
- Run calibration once the selected checkpoint is ready, then run held-out testing.
- Inspect errors and write supported observations in the report. Do not choose a
  new threshold by looking at test outcomes.

API reference used: https://docs.ultralytics.com/modes/predict/
