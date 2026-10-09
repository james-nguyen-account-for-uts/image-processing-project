"""Validation-selected geometry baselines using supplied YOLO person boxes.

The evaluation unit is an annotated person region, not an image. No person
localisation is performed, and no-person/empty images are never counted as TNs.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version
import yaml


@dataclass(frozen=True)
class GeometrySettings:
    data_yaml: str = "data/fall/data.yaml"
    output_dir: str = "runs/geometry"
    fall_class: str = "fall"
    upright_class: str = "up"
    no_person_class: str = "0 person"
    timing_repeats: int = 25

    def validate(self) -> None:
        if type(self.timing_repeats) is not int or self.timing_repeats < 1:
            raise ValueError("timing_repeats must be a positive integer")
        for field in ("data_yaml", "output_dir", "fall_class", "upright_class", "no_person_class"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"{field} must be a non-empty string")
        if len({self.fall_class, self.upright_class, self.no_person_class}) != 3:
            raise ValueError("Class names must be distinct")


@dataclass(frozen=True)
class Region:
    image: str
    region_id: int
    original_class: str
    true_fall: bool
    image_width: int
    image_height: int
    center_x: float
    center_y: float
    box_width: float
    box_height: float

    @property
    def aspect_ratio(self) -> float:
        # YOLO w/h are normalised against different image dimensions.
        return (self.box_width * self.image_width) / (self.box_height * self.image_height)


@dataclass
class DatasetSplit:
    name: str
    regions: list[Region]
    excluded: list[dict]
    image_count: int
    class_counts: dict[str, int]
    fingerprint: str

    def summary(self) -> dict:
        return {
            "images": self.image_count,
            "evaluated_images": len({r.image for r in self.regions}),
            "evaluated_person_regions": len(self.regions),
            "fall_regions": sum(r.true_fall for r in self.regions),
            "up_regions": sum(not r.true_fall for r in self.regions),
            "annotation_instances_by_class": self.class_counts,
            "excluded_records": len(self.excluded),
            "geometry_input_sha256": self.fingerprint,
        }


@dataclass(frozen=True)
class GeometryRule:
    name: str
    aspect_threshold: float
    position_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"aspect_ratio", "aspect_ratio_position"}:
            raise ValueError(f"Unknown geometry rule: {self.name}")
        if not math.isfinite(self.aspect_threshold) or self.aspect_threshold < 0:
            raise ValueError("aspect_threshold must be finite and non-negative")
        if not math.isfinite(self.position_threshold) or not 0 <= self.position_threshold <= 1:
            raise ValueError("position_threshold must be in [0, 1]")
        if self.name == "aspect_ratio" and self.position_threshold != 0:
            raise ValueError("The aspect-only rule cannot have a position threshold")

    def predict(self, regions: list[Region]) -> np.ndarray:
        return np.fromiter(
            (r.aspect_ratio > self.aspect_threshold and r.center_y >= self.position_threshold
             for r in regions), dtype=bool, count=len(regions),
        )


def load_config(path: str | Path) -> GeometrySettings:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Geometry configuration must be a YAML mapping")
    settings = GeometrySettings(**values)
    settings.validate()
    return settings


def load_split(data_yaml: str | Path, split: str, settings: GeometrySettings) -> DatasetSplit:
    """Read image dimensions and labels; reject missing or malformed annotations."""
    yaml_path = Path(data_yaml).resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    expected = {settings.fall_class, settings.upright_class, settings.no_person_class}
    if set(names.values()) != expected or len(names) != 3:
        raise ValueError(f"Expected exactly the configured classes {sorted(expected)}, got {names}")
    root = Path(data.get("path") or ".")
    if not root.is_absolute():
        root = yaml_path.parent / root
    split_path = data.get(split)
    if not isinstance(split_path, str):
        raise ValueError(f"Dataset split '{split}' must name an images directory")
    image_dir = Path(split_path)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    image_dir = image_dir.resolve()
    if not image_dir.is_dir() or image_dir.name != "images":
        raise ValueError(f"Expected an existing .../images directory for {split}: {image_dir}")
    label_dir = image_dir.parent / "labels"
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    if not images:
        raise ValueError(f"No images found for {split}: {image_dir}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {image_dir}; label ownership is ambiguous")
    regions, excluded = [], []
    class_counts = Counter({name: 0 for name in names.values()})
    fingerprint = hashlib.sha256()
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {label_path}; use an empty file for no boxes")
        label_bytes = label_path.read_bytes()
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        image_id = image_path.relative_to(root.resolve()).as_posix()
        fingerprint.update(json.dumps([image_id, image_width, image_height], ensure_ascii=False).encode())
        fingerprint.update(label_bytes)
        lines = label_bytes.decode("utf-8-sig").splitlines()
        if not any(line.strip() for line in lines):
            excluded.append({"image": image_id, "region_id": "", "original_class": "",
                             "reason": "empty_annotation_no_person_region"})
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError("expected class_id cx cy width height")
                class_id = int(parts[0])
                cx, cy, width, height = map(float, parts[1:])
                if class_id not in names:
                    raise ValueError(f"unknown class_id {class_id}")
                if not all(math.isfinite(v) for v in (cx, cy, width, height)):
                    raise ValueError("box coordinates must be finite")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError("expected normalised centres in [0, 1] and sizes in (0, 1]")
            except ValueError as error:
                raise ValueError(f"{label_path}:{line_number}: {error}") from error
            class_name = names[class_id]
            class_counts[class_name] += 1
            if class_name == settings.no_person_class:
                excluded.append({"image": image_id, "region_id": line_number,
                                 "original_class": class_name, "reason": "no_person_class_not_a_person_region"})
                continue
            regions.append(Region(image_id, line_number, class_name, class_name == settings.fall_class,
                                  image_width, image_height, cx, cy, width, height))
    return DatasetSplit(split, regions, excluded, len(images), dict(class_counts), fingerprint.hexdigest())


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = np.asarray(truth, dtype=bool), np.asarray(prediction, dtype=bool)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("Expected matching one-dimensional truth and prediction arrays")
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    tn = int(np.count_nonzero(~truth & ~prediction))
    fn = int(np.count_nonzero(truth & ~prediction))
    return {
        "sample_count": len(truth), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fall_precision": tp / (tp + fp) if tp + fp else 0.0,
        "fall_recall": tp / (tp + fn) if tp + fn else 0.0,
        "fall_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "up_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "no_person_false_positive_rate": None,
        "confusion_matrix": {"labels": ["No Fall", "Fall"], "rows": "true", "columns": "predicted",
                             "counts": [[tn, fp], [fn, tp]]},
    }


def tune_rules(validation: list[Region]) -> tuple[list[GeometryRule], dict]:
    """Search every distinct validation decision, efficiently via suffix sums.

    Thresholds are 0 plus observed ratios, and 0/1 plus observed vertical
    centres. Ties: higher Fall recall, then lower position threshold (less
    dependence on camera position), then lower aspect threshold.
    """
    if not validation or len({r.true_fall for r in validation}) != 2:
        raise ValueError("Threshold selection needs both fall and up validation regions")
    ratio = np.array([r.aspect_ratio for r in validation])
    position = np.array([r.center_y for r in validation])
    truth = np.array([r.true_fall for r in validation])
    ratio_thresholds = np.unique(np.r_[0.0, ratio])
    position_thresholds = np.unique(np.r_[0.0, position, 1.0])
    shape = (len(ratio_thresholds), len(position_thresholds))
    ratio_bin = np.searchsorted(ratio_thresholds, ratio)
    position_bin = np.searchsorted(position_thresholds, position)

    def counts(mask: np.ndarray) -> np.ndarray:
        histogram = np.zeros(shape, dtype=np.int64)
        np.add.at(histogram, (ratio_bin[mask], position_bin[mask]), 1)
        suffix = histogram[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
        # Strict r > threshold, but inclusive cy >= threshold.
        return np.vstack((suffix[1:], np.zeros((1, shape[1]), dtype=np.int64)))

    tp, fp = counts(truth), counts(~truth)
    positive_count = int(truth.sum())
    denominator = positive_count + tp + fp
    f1 = np.divide(2 * tp, denominator, out=np.zeros(shape), where=denominator > 0)
    recall = tp / positive_count

    def choose(name: str, columns: int) -> GeometryRule:
        scores = f1[:, :columns]
        best = scores == scores.max()
        best &= recall[:, :columns] == recall[:, :columns][best].max()
        indices = np.argwhere(best)
        i, j = min(indices, key=lambda ij: (position_thresholds[ij[1]], ratio_thresholds[ij[0]]))
        return GeometryRule(name, float(ratio_thresholds[i]), float(position_thresholds[j]))

    return [choose("aspect_ratio", 1), choose("aspect_ratio_position", shape[1])], {
        "selection_split": "val", "primary_metric": "fall_f1", "tie_break_metric": "fall_recall",
        "further_ties": "lower position threshold, then lower aspect threshold",
        "aspect_candidates": shape[0], "position_candidates": shape[1],
        "joint_candidates": shape[0] * shape[1],
        "search": "All distinct validation decisions using observed feature boundaries plus 0/1 endpoints",
        "position_hypothesis": "A fall-like box centre is at or below a normalised vertical threshold",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_rule(rule: GeometryRule, split: DatasetSplit, output_dir: Path, repeats: int) -> dict:
    truth = np.array([r.true_fall for r in split.regions], dtype=bool)
    prediction = rule.predict(split.regions)
    metrics = classification_metrics(truth, prediction)
    rows = []
    for region, predicted in zip(split.regions, prediction):
        row = asdict(region)
        row.update({"split": split.name, "rule": rule.name, "aspect_ratio": region.aspect_ratio,
                    "true_label": "Fall" if region.true_fall else "No Fall",
                    "predicted_label": "Fall" if predicted else "No Fall",
                    "outcome": ("TP" if predicted else "FN") if region.true_fall else ("FP" if predicted else "TN")})
        rows.append(row)
    fields = ["split", "rule", "image", "region_id", "original_class", "true_fall", "true_label",
              "predicted_label", "outcome", "image_width", "image_height", "center_x", "center_y",
              "box_width", "box_height", "aspect_ratio"]
    prefix = f"{split.name}_{rule.name}"
    write_csv(output_dir / f"{prefix}_predictions.csv", rows, fields)
    write_csv(output_dir / f"{prefix}_errors.csv", [r for r in rows if r["outcome"] in {"FP", "FN"}], fields)
    matrix = metrics["confusion_matrix"]["counts"]
    write_csv(output_dir / f"{prefix}_confusion.csv",
              [{"true_label": label, "predicted_no_fall": values[0], "predicted_fall": values[1]}
               for label, values in zip(["No Fall", "Fall"], matrix)],
              ["true_label", "predicted_no_fall", "predicted_fall"])
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        rule.predict(split.regions)
        durations.append(time.perf_counter_ns() - start)
    metrics["timing"] = {
        "median_total_ms": statistics.median(durations) / 1e6,
        "median_ms_per_person_region": statistics.median(durations) / 1e6 / len(split.regions)
        if split.regions else None,
        "repeats": repeats, "warmup_passes": 1,
        "includes": "pixel aspect ratio calculation, threshold comparisons, prediction array construction",
        "excludes": "image/label loading, person localisation, threshold search, metric calculation, CSV writing",
    }
    return metrics


def run_experiment(config_path: str | Path, model_path: str | Path | None = None) -> dict:
    settings = load_config(config_path)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = load_split(settings.data_yaml, "val", settings)
    if model_path is None:
        rules, selection = tune_rules(validation.regions)
        validation_metrics = {r.name: classification_metrics(
            np.array([x.true_fall for x in validation.regions]), r.predict(validation.regions)) for r in rules}
        selected_rule = max(rules, key=lambda r: (
            validation_metrics[r.name]["fall_f1"], validation_metrics[r.name]["fall_recall"],
            r.name == "aspect_ratio"))
        model = {"schema_version": 1, "evaluation_unit": "annotated_person_region",
                 "class_mapping": {"fall": settings.fall_class, "up": settings.upright_class,
                                   "excluded_no_person": settings.no_person_class},
                 "rules": [asdict(r) for r in rules], "selection": selection,
                 "selected_rule_by_validation": selected_rule.name,
                 "validation_input_sha256": validation.fingerprint}
    else:
        model = json.loads(Path(model_path).read_text(encoding="utf-8"))
        expected_mapping = {"fall": settings.fall_class, "up": settings.upright_class,
                            "excluded_no_person": settings.no_person_class}
        if model.get("schema_version") != 1 or model.get("class_mapping") != expected_mapping:
            raise ValueError("Saved rule schema/class mapping does not match this experiment")
        rules = [GeometryRule(**r) for r in model["rules"]]
        if {r.name for r in rules} != {"aspect_ratio", "aspect_ratio_position"} or len(rules) != 2:
            raise ValueError("Saved model must contain both geometry variants")
        if model.get("selected_rule_by_validation") not in {r.name for r in rules}:
            raise ValueError("Saved selected rule is invalid")
    # Freeze both thresholds and the recommended variant before opening test inputs.
    (output_dir / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test = load_split(settings.data_yaml, "test", settings)
    results = {
        "evaluation_unit": "annotated_person_region", "input": "ground-truth supplied person boxes",
        "settings": asdict(settings), "model": model,
        "reused_model": str(model_path) if model_path is not None else None,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "numpy": np.__version__,
                        "pillow": pillow_version, "pyyaml": yaml.__version__},
        "dataset_yaml_sha256": hashlib.sha256(Path(settings.data_yaml).read_bytes()).hexdigest(),
        "splits": {},
        "limitations": [
            "Supplied person boxes remove localisation errors; these scores are not full-image detection scores.",
            "0 person annotations and empty-label images are excluded, not counted as true negatives.",
            "No-person false-positive rate is undefined for this person-region baseline.",
            "Position thresholds depend on camera framing and do not demonstrate generalisation to other views.",
            "This experiment does not audit cross-split duplicate or same-source frames.",
        ],
    }
    for split in (validation, test):
        write_csv(output_dir / f"{split.name}_excluded.csv", split.excluded,
                  ["image", "region_id", "original_class", "reason"])
        results["splits"][split.name] = {"dataset": split.summary(), "rules": {
            rule.name: evaluate_rule(rule, split, output_dir, settings.timing_repeats) for rule in rules}}
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Bounding-box geometry experiment", "",
             "Evaluation uses annotated person regions. Thresholds and the recommended variant are selected on validation only.",
             "", "Fall: r > aspect_threshold. The position variant also requires cy >= position_threshold.", "",
             f"Recommended variant from validation: `{model['selected_rule_by_validation']}`.", "",
             "| Split | Rule | r threshold | cy threshold | Regions | Precision | Recall | F1 | Up FPR |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in (validation, test):
        for rule in rules:
            m = results["splits"][split.name]["rules"][rule.name]
            fpr = "N/A" if m["up_false_positive_rate"] is None else f"{m['up_false_positive_rate']:.4f}"
            lines.append(f"| {split.name} | {rule.name} | {rule.aspect_threshold:.9g} | {rule.position_threshold:.9g} | "
                         f"{m['sample_count']} | {m['fall_precision']:.4f} | {m['fall_recall']:.4f} | {m['fall_f1']:.4f} | {fpr} |")
    lines.extend(["", "See metrics.json for TP/FP/TN/FN, timing, environment and input fingerprints; "
                  "*_predictions.csv for each region; *_errors.csv for false positives and missed falls; "
                  "*_excluded.csv for excluded records.", "", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in results["limitations"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and evaluate bounding-box geometry fall rules")
    parser.add_argument("--config", default="configs/geometry.yaml")
    parser.add_argument("--model", help="Reuse frozen model.json thresholds instead of selecting them again")
    args = parser.parse_args()
    results = run_experiment(args.config, args.model)
    for split, result in results["splits"].items():
        for name, metrics in result["rules"].items():
            print(f"{split} {name}: regions={metrics['sample_count']}, precision={metrics['fall_precision']:.4f}, "
                  f"recall={metrics['fall_recall']:.4f}, F1={metrics['fall_f1']:.4f}")
    print(f"Results saved to {results['settings']['output_dir']}")


if __name__ == "__main__":
    main()
