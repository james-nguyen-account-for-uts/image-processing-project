"""Shared YOLO-annotation reader and person-region metrics."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


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


def parse_yolo_box(line: str, path: Path, line_number: int, names: dict) -> tuple:
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
        raise ValueError(f"{path}:{line_number}: {error}") from error
    return class_id, cx, cy, width, height


def load_split(data_yaml: str | Path, split: str, settings) -> DatasetSplit:
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
            class_id, cx, cy, width, height = parse_yolo_box(line, label_path, line_number, names)
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

