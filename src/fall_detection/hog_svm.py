"""HOG + linear SVM fall classification of supplied, annotated person regions.

Training fits preprocessing and the classifier on train only. Validation selects
C and a Fall decision threshold. Test evaluation never refits either component.
This method does not locate people and excludes no-person annotations.
"""

import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage import __version__ as skimage_version
from skimage.feature import hog
from sklearn import __version__ as sklearn_version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import yaml

from .geometry import (GeometrySettings, Region, classification_metrics,
                       load_split, parse_yolo_box, regions_from_boxes)


@dataclass(frozen=True)
class HogSettings:
    data_yaml: str = "data/fall/data.yaml"
    output_dir: str = "runs/hog_svm"
    fall_class: str = "fall"
    upright_class: str = "up"
    no_person_class: str = "0 person"
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    random_seed: int = 42

    def validate(self) -> None:
        for name in ("data_yaml", "output_dir", "fall_class", "upright_class", "no_person_class"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len({self.fall_class, self.upright_class, self.no_person_class}) != 3:
            raise ValueError("Class names must be distinct")
        if not self.c_values or any(isinstance(c, bool) or not isinstance(c, (float, int))
                                    or not math.isfinite(c) or c <= 0 for c in self.c_values):
            raise ValueError("c_values must contain positive finite numbers")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")


def load_config(path: str | Path) -> HogSettings:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("HOG-SVM configuration must be a YAML mapping")
    if "c_values" in values:
        if not isinstance(values["c_values"], (list, tuple)):
            raise ValueError("c_values must be a sequence")
        values["c_values"] = tuple(values["c_values"])
    settings = HogSettings(**values)
    settings.validate()
    return settings


def geometry_settings(settings: HogSettings) -> GeometrySettings:
    return GeometrySettings(data_yaml=settings.data_yaml, output_dir=settings.output_dir,
                            fall_class=settings.fall_class, upright_class=settings.upright_class,
                            no_person_class=settings.no_person_class)


def dataset_root(data_yaml: str | Path) -> Path:
    yaml_path = Path(data_yaml).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data.get("path") or ".")
    return (root if root.is_absolute() else yaml_path.parent / root).resolve()


def preprocess_region(image: Image.Image, region: Region) -> np.ndarray:
    """Crop a supplied YOLO box, preserve its aspect ratio, and pad to 128²."""
    width, height = image.size
    if (width, height) != (region.image_width, region.image_height):
        raise ValueError(f"Image dimensions changed for {region.image}")
    x1 = max(0, math.floor((region.center_x - region.box_width / 2) * width))
    y1 = max(0, math.floor((region.center_y - region.box_height / 2) * height))
    x2 = min(width, math.ceil((region.center_x + region.box_width / 2) * width))
    y2 = min(height, math.ceil((region.center_y + region.box_height / 2) * height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid person crop in {region.image}, region {region.region_id}")
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    scale = min(128 / crop.width, 128 / crop.height)
    new_size = (min(128, max(1, round(crop.width * scale))),
                min(128, max(1, round(crop.height * scale))))
    crop = crop.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (128, 128), color=0)
    canvas.paste(crop, ((128 - new_size[0]) // 2, (128 - new_size[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def hog_features(image: Image.Image, region: Region) -> np.ndarray:
    return hog(preprocess_region(image, region), orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def features_for_regions(regions: list[Region], root: Path) -> np.ndarray:
    if not regions:
        raise ValueError("No Fall/Up person regions available for HOG-SVM")
    features = []
    # Consecutive annotations usually belong to the same image; reuse its decode.
    previous_path, image = None, None
    try:
        for region in regions:
            path = root / region.image
            if path != previous_path:
                if image is not None:
                    image.close()
                image = Image.open(path)
                previous_path = path
            features.append(hog_features(image, region))
    finally:
        if image is not None:
            image.close()
    return np.stack(features)


def choose_threshold(scores: np.ndarray, truth: np.ndarray) -> tuple[float, dict]:
    """Select Fall F1, then recall, using validation scores only."""
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if scores.ndim != 1 or scores.shape != truth.shape or len(scores) == 0:
        raise ValueError("Expected matching non-empty one-dimensional scores and labels")
    if not np.isfinite(scores).all() or len(np.unique(truth)) != 2:
        raise ValueError("Validation needs finite scores and both Fall and Up regions")
    candidates = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores),
                       np.nextafter(scores.max(), np.inf)]
    best = None
    for threshold in candidates:
        metrics = classification_metrics(truth, scores >= threshold)
        key = (metrics["fall_f1"], metrics["fall_recall"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(split, scores: np.ndarray, threshold: float, settings: HogSettings,
                   model_path: Path, output_dir: Path, elapsed: float | None, stage: str,
                   feature_hash: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = np.array([r.true_fall for r in split.regions], dtype=bool)
    prediction = scores >= threshold
    metrics = classification_metrics(truth, prediction)
    fields = ["split", "image", "region_id", "original_class", "true_label",
              "predicted_label", "outcome", "decision_score", "decision_threshold"]
    rows = []
    for region, score, predicted in zip(split.regions, scores, prediction):
        outcome = ("TP" if predicted else "FN") if region.true_fall else ("FP" if predicted else "TN")
        rows.append({"split": split.name, "image": region.image, "region_id": region.region_id,
                     "original_class": region.original_class,
                     "true_label": "Fall" if region.true_fall else "No Fall",
                     "predicted_label": "Fall" if predicted else "No Fall", "outcome": outcome,
                     "decision_score": float(score), "decision_threshold": threshold})
    _write_csv(output_dir / f"{split.name}_predictions.csv", rows, fields)
    _write_csv(output_dir / f"{split.name}_errors.csv", [r for r in rows if r["outcome"] in {"FP", "FN"}], fields)
    _write_csv(output_dir / f"{split.name}_excluded.csv", split.excluded,
               ["image", "region_id", "original_class", "reason"])
    matrix = metrics["confusion_matrix"]["counts"]
    _write_csv(output_dir / f"{split.name}_confusion.csv",
               [{"true_label": label, "predicted_no_fall": counts[0], "predicted_fall": counts[1]}
                for label, counts in zip(["No Fall", "Fall"], matrix)],
               ["true_label", "predicted_no_fall", "predicted_fall"])
    result = {
        "method": "hog_svm", "stage": stage, "split": split.name,
        "evaluation_unit": "annotated_person_region", "input": "ground-truth supplied person boxes",
        "model_path": str(model_path), "output_dir": str(output_dir), "settings": asdict(settings),
        "dataset": split.summary(), "feature_sha256": feature_hash, "metrics": metrics,
        "timing": None if elapsed is None else {
            "total_seconds": elapsed, "seconds_per_region": elapsed / len(split.regions),
            "includes": "image loading, crop, resize, HOG extraction, standardization, SVM scoring",
            "excludes": "person localisation, metric calculation, artifact writing"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_image": skimage_version, "scikit_learn": sklearn_version},
        "limitations": ["Supplied boxes remove person-localisation errors; compare separately with full-image YOLO.",
                        "No-person annotations and empty labels are excluded, not counted as true negatives.",
                        "Cross-split near-duplicates and source-video frames require a separate audit."],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _feature_hash(features: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()


def train_model(config_path: str | Path) -> dict:
    """Fit on train; select C and Fall threshold on val; never open test."""
    settings = load_config(config_path)
    geometry = geometry_settings(settings)
    train = load_split(settings.data_yaml, "train", geometry)
    validation = load_split(settings.data_yaml, "val", geometry)
    if len({r.true_fall for r in train.regions}) != 2:
        raise ValueError("Training needs both Fall and Up person regions")
    root = dataset_root(settings.data_yaml)
    train_x = features_for_regions(train.regions, root)
    val_x = features_for_regions(validation.regions, root)
    train_y = np.array([r.true_fall for r in train.regions], dtype=bool)
    val_y = np.array([r.true_fall for r in validation.regions], dtype=bool)
    candidates = []
    best = None
    for c in sorted(set(settings.c_values)):
        pipeline = make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced",
                                                             random_state=settings.random_seed, max_iter=10000))
        pipeline.fit(train_x, train_y)
        scores = pipeline.decision_function(val_x)
        threshold, metrics = choose_threshold(scores, val_y)
        candidates.append({"C": c, "threshold": threshold, "metrics": metrics})
        key = (metrics["fall_f1"], metrics["fall_recall"], -c, threshold)
        if best is None or key > best[0]:
            best = (key, pipeline, c, threshold, scores)
    _, pipeline, selected_c, threshold, scores = best
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    model = {"schema_version": 1, "method": "hog_svm", "evaluation_unit": "annotated_person_region",
             "class_mapping": {"fall": settings.fall_class, "up": settings.upright_class,
                               "excluded_no_person": settings.no_person_class},
             "pipeline": pipeline, "C": selected_c, "threshold": threshold,
             "feature_spec": {"crop": "YOLO supplied box clipped to image; aspect-preserving resize and black pad",
                              "size": [128, 128], "grayscale": True, "orientations": 9,
                              "pixels_per_cell": [8, 8], "cells_per_block": [2, 2], "block_norm": "L2-Hys"},
             "selection_split": "val", "selection_metric": "Fall F1, then Fall recall",
             "random_seed": settings.random_seed, "train_fingerprint": train.fingerprint,
             "validation_fingerprint": validation.fingerprint}
    joblib.dump(model, model_path)
    calibration = _write_results(validation, scores, threshold, settings, model_path,
                                 output_dir / "calibration", None, "calibration", _feature_hash(val_x))
    calibration["training_dataset"] = train.summary()
    calibration["training_feature_sha256"] = _feature_hash(train_x)
    calibration["selection_candidates"] = candidates
    calibration["selected_C"] = selected_c
    calibration["decision_threshold"] = threshold
    (output_dir / "calibration" / "metrics.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return calibration


def load_model(path: Path, settings: HogSettings) -> dict:
    model = joblib.load(path)
    expected = {"fall": settings.fall_class, "up": settings.upright_class,
                "excluded_no_person": settings.no_person_class}
    if (not isinstance(model, dict) or model.get("schema_version") != 1
            or model.get("method") != "hog_svm" or model.get("class_mapping") != expected):
        raise ValueError("Saved HOG-SVM model schema or class mapping does not match this experiment")
    return model


def evaluate_model(config_path: str | Path, model_path: str | Path | None = None,
                   split: str = "test") -> dict:
    """Score a frozen model on the requested split without reading train/val."""
    if split not in {"val", "test"}:
        raise ValueError("HOG-SVM evaluation split must be val or test")
    settings = load_config(config_path)
    model_path = Path(model_path) if model_path is not None else Path(settings.output_dir) / "model.joblib"
    model = load_model(model_path, settings)
    data = load_split(settings.data_yaml, split, geometry_settings(settings))
    start = time.perf_counter()
    features = features_for_regions(data.regions, dataset_root(settings.data_yaml))
    scores = model["pipeline"].decision_function(features)
    elapsed = time.perf_counter() - start
    result = _write_results(data, scores, model["threshold"], settings, model_path,
                            Path(settings.output_dir) / "evaluation" / split,
                            elapsed, "evaluation", _feature_hash(features))
    result["selected_C"] = model["C"]
    result["decision_threshold"] = model["threshold"]
    (Path(result["output_dir"]) / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def predict_image(config_path: str | Path, source: str | Path, model_path: str | Path | None = None,
                  boxes: list[list[float]] | None = None, labels: str | Path | None = None,
                  output: str | Path | None = None) -> dict:
    """Classify supplied person boxes in one image; never infer boxes or refit."""
    if (boxes is None) == (labels is None):
        raise ValueError("HOG-SVM needs either pixel boxes or a YOLO labels file")
    settings = load_config(config_path)
    model_path = Path(model_path) if model_path is not None else Path(settings.output_dir) / "model.joblib"
    model = load_model(model_path, settings)
    source = Path(source)
    with Image.open(source) as image:
        width, height = image.size
        excluded = []
        if boxes is not None:
            regions = regions_from_boxes(width, height, boxes, str(source))
        else:
            data = yaml.safe_load(Path(settings.data_yaml).read_text(encoding="utf-8"))
            names = data["names"]
            names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
            if set(names.values()) != {settings.fall_class, settings.upright_class, settings.no_person_class}:
                raise ValueError("Dataset class names do not match the saved model")
            regions = []
            label_path = Path(labels)
            for number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                class_id, cx, cy, bw, bh = parse_yolo_box(line, label_path, number, names)
                if names[class_id] == settings.no_person_class:
                    excluded.append({"region_id": number, "reason": "no_person_class_not_a_person_region"})
                    continue
                # The annotation class is deliberately discarded at inference.
                regions.append(Region(str(source), number, "", False, width, height, cx, cy, bw, bh))
        if regions:
            features = np.stack([hog_features(image, region) for region in regions])
            scores = model["pipeline"].decision_function(features)
        else:
            scores = np.array([])
    output_path = Path(output) if output is not None else Path(settings.output_dir) / "predictions" / f"{source.stem}.json"
    protected = {source.resolve(), model_path.resolve()}
    if labels is not None:
        protected.add(Path(labels).resolve())
    if output_path.resolve() in protected:
        raise ValueError("Prediction output must not overwrite the image, annotations or saved model")
    decisions = scores >= model["threshold"]
    result = {"method": "hog_svm", "source": str(source), "model_path": str(model_path),
              "input": "supplied_yolo_labels" if labels is not None else "supplied_pixel_boxes",
              "threshold": model["threshold"], "person_count": len(regions),
              "image_prediction": ("Fall" if decisions.any() else "No Fall") if regions else None,
              "regions": [{"region_id": region.region_id, "decision_score": float(score),
                           "prediction": "Fall" if decision else "No Fall"}
                          for region, score, decision in zip(regions, scores, decisions)],
              "excluded": excluded, "output_path": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
