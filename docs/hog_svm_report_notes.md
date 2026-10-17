# HOG-SVM contribution notes for Assessment Task 2

## Method and input

The HOG-SVM experiment classifies annotated person regions as Fall or No Fall
(Up). It uses the Roboflow fall dataset v1 YOLO train, validation, and test
splits. The class `0 person` and images with empty labels have no person crop,
so they are counted in the excluded-record CSV rather than as true negatives.
This is a person-region classification experiment: ground-truth boxes supply
the person locations, unlike the full-image YOLO experiment.

Each annotated Fall or Up box is clipped to the image boundary, cropped,
converted to grayscale, resized without changing aspect ratio, and centred
on a black 128 × 128 canvas. HOG extracts edge-orientation features using nine
orientation bins, 8 × 8 pixel cells, 2 × 2 cell blocks, and L2-Hys block
normalisation. A standard scaler and class-balanced linear SVM are fitted on
the training split only. Candidate regularisation strengths are specified in
`configs/hog_svm.yaml`. The validation split selects both C and a decision
threshold by maximising Fall F1; Fall recall breaks ties. These settings are
frozen before the test split is evaluated.

## Actual local experiment on Roboflow fall v1

The YOLOv8 export was downloaded on 17 October 2026 and extracted locally to
`data/fall/`. Its `data.yaml` used `../train/images`, `../valid/images`, and
`../test/images`; those three paths were corrected to point inside `data/fall/`.
The original ZIP was retained. No dataset files or model weights are committed.

| Split | Images | Fall regions | Up regions | Evaluated person regions |
|---|---:|---:|---:|---:|
| Train | 3,334 | 2,563 | 987 | 3,550 |
| Validation | 872 | 739 | 190 | 929 |
| Test | 446 | 366 | 105 | 471 |

The training split has one `0 person` annotation; there are none in validation
or test. Manual inspection showed that this one image actually contains a
visible person. Its annotated box was excluded rather than relabelled without
an agreed ground truth. It is not counted as a true negative.

The validation search compared C values 0.01, 0.1, 1, and 10, as proposed in
Assessment Task 1. C=0.01 gave the highest Fall F1 (0.98586) and recall
(0.99053). Each of the other three values gave Fall F1 0.98584 and recall
0.98917 with its selected threshold. The chosen decision threshold was
-0.7226809. The selected model converged in 551 of 10,000 permitted
iterations. Some unselected candidates emitted convergence warnings.

With those settings fixed, the model yielded these results on the official
test split:

| Metric | Test result |
|---|---:|
| Fall precision | 0.9681 |
| Fall recall | 0.9945 |
| Fall F1 | 0.9811 |
| True positives | 364 |
| False positives | 12 |
| True negatives | 93 |
| False negatives | 2 |
| Up false-positive rate | 12 / 105 = 0.1143 |

An earlier run used only C={0.1, 1, 10} and its test results were inspected
before the missing, pre-specified C=0.01 candidate was noticed. The corrected
model was still selected using validation data only, and its test confusion
counts did not change. The test set has been viewed more than once, so this
should not be described as a single blind test evaluation.

These are **person-region** scores with supplied ground-truth boxes. They
cannot be ranked directly against YOLO's full-image detection results. One
local evaluation pass took 3.09 seconds for 471 regions (6.56 ms per region),
including image loading, crop, resize, HOG extraction, standardisation and SVM
scoring. It excludes person localisation, model loading, metric calculation and
file writing, and is not a cross-hardware benchmark.

The exact records are in `runs/hog_svm/evaluation/test/metrics.json`,
`test_predictions.csv`, `test_errors.csv`, and `test_confusion.csv`. The model
and the validation record are in `runs/hog_svm/model.joblib` and
`runs/hog_svm/calibration/metrics.json`.

## Discussion points grounded in the method

HOG describes visible edges and body shape but does not capture the motion
leading to a fall. Sitting, bending, lying down, camera angle, occlusion, and
background inside the crop may therefore cause errors. The two missed Fall
examples include a person crouched or seated on stairs and a noisy frame with
an apparently upright person; the latter illustrates possible single-frame
label ambiguity rather than proving that the annotation is wrong. False
alarms include an upright person viewed from an oblique camera and an image
rotated sideways with two upright people. These are examples from manual
inspection, not a quantified error taxonomy. Supplying the box
removes localisation errors, so a favourable HOG-SVM region score would not
establish better full-image fall detection than YOLO. The dataset's `0 person`
labels need manual inspection because a box with that class can still surround
a visible person; such annotations are excluded by this experiment.

There is also a split-independence concern. Image file bytes had no exact
cross-split duplicates, but the source name before Roboflow's `.rf.` suffix
appeared in both train and validation for 254 names and in train and test for
169 names. Of these, 139 and 82 name matches, respectively, had mean absolute
RGB pixel differences below 5/255 after resizing both images to 64 × 64.
This screening indicates near-identical images across the official splits.
Other sequential video frames may also span splits. The reported test score
therefore describes this dataset partition and may overstate performance on
independent scenes. A future experiment should group original images or video
sources before splitting, then repeat training and threshold selection. Do
not present this score as evidence for real-world fall-monitoring reliability.

## Reproducibility and code appendix

- Implementation: `src/fall_detection/hog_svm.py`
- Shared command integration: `src/fall_detection/train.py`, `evaluate.py`,
  `detect.py`, and `config.py`
- Configuration: `configs/hog_svm.yaml`
- Behaviour checks: `tests/test_hog_svm.py`
- Saved model: `runs/hog_svm/model.joblib` (local, intentionally not committed)
- Per-region predictions, errors, exclusions, and confusion counts:
  `runs/hog_svm/evaluation/test/`
