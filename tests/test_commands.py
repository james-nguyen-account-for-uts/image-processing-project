import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from fall_detection import detect, evaluate, geometry, train
from fall_detection.config import resolve_config


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    for split in ["valid", "test"]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for name, label in [("fall", "1 0.5 0.7 0.4 0.2\n"), ("up", "2 0.5 0.3 0.05 0.4\n")]:
            Image.new("RGB", (200, 100)).save(root / split / "images" / f"{name}.png")
            (root / split / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("val: valid/images\ntest: test/images\nnames: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    config = tmp_path / "geometry.yaml"
    output = tmp_path / "geometry"
    config.write_text(f"data_yaml: '{data_yaml.as_posix()}'\noutput_dir: '{output.as_posix()}'\n"
                      "timing_repeats: 2\n", encoding="utf-8")
    return root, data_yaml, config, output


def test_geometry_train_reads_validation_only_without_ultralytics(dataset, monkeypatch):
    root, data_yaml, config, output = dataset
    # Calibration must work without a test split, and without a YOLO installation.
    data_yaml.write_text("val: valid/images\nnames: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setattr(sys, "argv", ["fall-train", "--method", "geometry", "--config", str(config)])
    train.main()
    model = json.loads((output / "model.json").read_text(encoding="utf-8"))
    assert model["selection"]["selection_split"] == "val"
    metrics = json.loads((output / "calibration/metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["splits"]) == {"val"}
    assert metrics["splits"]["val"]["rules"]["aspect_ratio"]["fall_f1"] == 1


def test_geometry_evaluate_does_not_tune_read_validation_or_modify_model(dataset, monkeypatch):
    root, data_yaml, config, output = dataset
    train.train(str(config), method="geometry")
    model = output / "model.json"
    frozen = model.read_bytes()
    # Evaluation needs only the requested test split, even with no validation path.
    data_yaml.write_text("test: test/images\nnames: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    (root / "test/labels/fall.txt").write_text("1 0.5 0.5 0.01 0.8\n", encoding="utf-8")

    def no_tuning(*args):
        pytest.fail("Evaluation must use frozen thresholds")

    monkeypatch.setattr(geometry, "tune_rules", no_tuning)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.setattr(sys, "argv", ["fall-evaluate", "--method", "geometry", "--config", str(config),
                                    "--model", str(model)])
    evaluate.main()
    assert model.read_bytes() == frozen
    metrics = json.loads((output / "evaluation/test/metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["splits"]) == {"test"}
    assert metrics["splits"]["test"]["rules"]["aspect_ratio"]["fn"] == 1
    assert (output / "evaluation/test/test_aspect_ratio_errors.csv").is_file()


def test_geometry_detection_only_applies_rules_and_aggregates_person_boxes(dataset, monkeypatch, capsys):
    root, _, config, output = dataset
    train.train(str(config), method="geometry")
    model = output / "model.json"
    frozen = model.read_bytes()

    def no_dataset(*args):
        pytest.fail("Detection must not load dataset splits or tune thresholds")

    monkeypatch.setattr(geometry, "load_split", no_dataset)
    monkeypatch.setattr(geometry, "tune_rules", no_dataset)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    image = root / "test/images/fall.png"
    result_path = output / "custom.json"
    monkeypatch.setattr(sys, "argv", ["fall-detect", "--method", "geometry", "--config", str(config),
                                    "--weights", str(model), "--source", str(image),
                                    "--bbox", "0", "0", "10", "90",
                                    "--bbox", "20", "20", "180", "70", "--output", str(result_path)])
    detect.main()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert [r["prediction"] for r in result["regions"]] == ["No Fall", "Fall"]
    assert result["regions"][1]["aspect_ratio"] == pytest.approx(3.2)
    assert result["image_prediction"] == "Fall"
    assert model.read_bytes() == frozen
    assert "Image: Fall" in capsys.readouterr().out


def test_geometry_annotation_ids_use_names_and_not_ground_truth_labels(dataset):
    root, data_yaml, config, output = dataset
    train.train(str(config), method="geometry")
    data_yaml.write_text("names: ['fall', '0 person', 'up']\n", encoding="utf-8")
    annotations = root / "source.txt"
    image = root / "test/images/fall.png"
    annotations.write_text("0 0.5 0.5 0.7 0.4\n1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    first = detect.detect(str(config), str(image), method="geometry", labels=str(annotations))
    annotations.write_text("2 0.5 0.5 0.7 0.4\n1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    second = detect.detect(str(config), str(image), method="geometry", labels=str(annotations))
    assert first["regions"] == second["regions"]
    assert first["person_count"] == 1
    assert first["excluded"][0]["region_id"] == 2


def test_geometry_empty_regions_are_not_reported_as_no_fall(dataset):
    root, _, config, _ = dataset
    train.train(str(config), method="geometry")
    annotations = root / "empty.txt"
    annotations.write_text("", encoding="utf-8")
    result = detect.detect(str(config), str(root / "test/images/fall.png"),
                           method="geometry", labels=str(annotations))
    assert result["image_prediction"] is None
    assert result["person_count"] == 0


@pytest.mark.parametrize("box", [[0, 0, 0, 50], [-1, 0, 20, 50], [0, 0, 201, 50],
                                  [0, 0, float("nan"), 50], [0, 0, 20]])
def test_geometry_detection_rejects_invalid_boxes(dataset, box):
    root, _, config, _ = dataset
    train.train(str(config), method="geometry")
    with pytest.raises(ValueError, match="Box 1"):
        detect.detect(str(config), str(root / "test/images/fall.png"), method="geometry", boxes=[box])


def test_geometry_detection_cannot_overwrite_saved_rules(dataset):
    root, _, config, output = dataset
    train.train(str(config), method="geometry")
    model = output / "model.json"
    original = model.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        detect.detect(str(config), str(root / "test/images/fall.png"), method="geometry",
                      boxes=[[0, 0, 100, 50]], output=str(model))
    assert original == model.read_bytes()


@pytest.mark.parametrize("argv", [["fall-detect", "--method", "geometry", "--source", "photo.jpg"],
                                  ["fall-detect", "--source", "photo.jpg", "--bbox", "0", "0", "10", "10"]])
def test_detection_cli_rejects_wrong_method_options(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        detect.main()
    assert error.value.code == 2


@pytest.mark.parametrize("method,expected", [("yolo", "configs/default.yaml"), ("geometry", "configs/geometry.yaml")])
def test_method_specific_default_configs(method, expected):
    assert resolve_config(method, None) == expected
    assert resolve_config(method, "custom.yaml") == "custom.yaml"


def test_yolo_default_training_evaluation_and_detection_are_preserved(tmp_path, monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, model):
            calls.append(("load", model))

        def train(self, **kwargs):
            calls.append(("train", kwargs))

        def val(self, **kwargs):
            calls.append(("val", kwargs))
            return types.SimpleNamespace(box=types.SimpleNamespace(map50=0.9, map=0.7))

        def predict(self, **kwargs):
            calls.append(("predict", kwargs))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    data = tmp_path / "data.yaml"
    data.write_text("{}", encoding="utf-8")
    config = tmp_path / "yolo.yaml"
    config.write_text(f"data_yaml: '{data.as_posix()}'\nproject_dir: '{tmp_path.as_posix()}'\n", encoding="utf-8")
    weights = tmp_path / "train/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"test weights")
    train.train(str(config))
    evaluate.evaluate(str(config))
    detect.detect(str(config), "0")
    assert calls[0] == ("load", "yolov8n.pt")
    assert calls[1][1]["epochs"] == 30 and calls[1][1]["device"] is None
    assert calls[3] == ("val", {"data": data.as_posix()})
    assert calls[5][1]["source"] == 0
    assert calls[5][1]["conf"] == 0.25
    assert calls[5][1]["show"] is True and calls[5][1]["save"] is True
    evaluate.evaluate(str(config), split="test")
    assert calls[-1] == ("val", {"data": data.as_posix(), "split": "test"})
