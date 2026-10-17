import csv

import numpy as np
from PIL import Image, ImageDraw
import pytest

from fall_detection import detect, evaluate, train
from fall_detection.hog_svm import (choose_threshold, load_config,
                                    preprocess_region, train_model)
from fall_detection.geometry import Region


def make_image(path, kind):
    image = Image.new("RGB", (96, 96), "black")
    draw = ImageDraw.Draw(image)
    if kind == "fall":
        for y in range(20, 80, 6):
            draw.line((10, y, 86, y), fill="white", width=2)
    elif kind == "up":
        for x in range(20, 80, 6):
            draw.line((x, 10, x, 86), fill="white", width=2)
    image.save(path)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    for split, count in [("train", 4), ("valid", 2), ("test", 2)]:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir()
        for kind, class_id in [("fall", 1), ("up", 2)]:
            for index in range(count):
                stem = f"{kind}_{index}"
                make_image(image_dir / f"{stem}.png", kind)
                (label_dir / f"{stem}.txt").write_text(
                    f"{class_id} 0.5 0.5 0.95 0.95\n", encoding="utf-8")
        make_image(image_dir / "empty.png", "empty")
        (label_dir / "empty.txt").write_text("0 0.5 0.5 0.95 0.95\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("train: train/images\nval: valid/images\ntest: test/images\n"
                         "names: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    config = tmp_path / "hog.yaml"
    output = tmp_path / "results"
    config.write_text(f"data_yaml: '{data_yaml.as_posix()}'\n"
                      f"output_dir: '{output.as_posix()}'\n"
                      "c_values: [0.1, 1.0]\n", encoding="utf-8")
    return root, data_yaml, config, output


def test_preserved_aspect_ratio_and_hog_shape():
    image = Image.new("RGB", (100, 50), "white")
    region = Region("sample", 1, "fall", True, 100, 50, 0.5, 0.5, 0.8, 0.8)
    canvas = preprocess_region(image, region)
    assert canvas.shape == (128, 128)
    assert np.count_nonzero(canvas[0]) == 0
    assert np.count_nonzero(canvas[64]) > 0


def test_threshold_selection_prioritises_recall_on_f1_tie():
    # Predict all Fall: TP=2, FP=4. Predict top two: TP=1, FP=1.
    scores = np.array([0.9, 0.1, 0.9, 0.7, 0.5, 0.3])
    truth = np.array([True, True, False, False, False, False])
    threshold, metrics = choose_threshold(scores, truth)
    assert metrics["fall_f1"] == pytest.approx(0.5)
    assert metrics["fall_recall"] == 1.0
    assert threshold <= 0.1


def test_training_uses_train_and_validation_only(dataset):
    root, data_yaml, config, output = dataset
    # The test split need not exist until final evaluation.
    data_yaml.write_text("train: train/images\nval: valid/images\n"
                         "names: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    result = train_model(config)
    assert result["training_dataset"]["evaluated_person_regions"] == 8
    assert result["dataset"]["evaluated_person_regions"] == 4
    assert result["metrics"]["fall_f1"] == 1.0
    assert result["metrics"]["sample_count"] == 4
    assert (output / "model.joblib").is_file()
    excluded = list(csv.DictReader((output / "calibration/val_excluded.csv").open(encoding="utf-8")))
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "no_person_class_not_a_person_region"
    assert load_config(config).c_values == (0.1, 1.0)


def test_frozen_test_evaluation_and_label_independent_inference(dataset, monkeypatch):
    root, data_yaml, config, output = dataset
    train.train(str(config), method="hog_svm")
    model_path = output / "model.joblib"
    frozen = model_path.read_bytes()
    # Deliberately change test labels; evaluation must reflect them, not recalibrate.
    (root / "test/labels/fall_0.txt").write_text("2 0.5 0.5 0.95 0.95\n", encoding="utf-8")
    data_yaml.write_text("test: test/images\nnames: ['0 person', 'fall', 'up']\n", encoding="utf-8")

    def no_train(*args, **kwargs):
        pytest.fail("Test evaluation must not train")

    monkeypatch.setattr("fall_detection.hog_svm.train_model", no_train)
    result = evaluate.evaluate(str(config), method="hog_svm")
    assert result["metrics"]["sample_count"] == 4
    assert result["metrics"]["fp"] == 1
    assert model_path.read_bytes() == frozen
    assert (output / "evaluation/test/test_errors.csv").is_file()

    image = root / "test/images/up_0.png"
    labels = root / "inference.txt"
    labels.write_text("1 0.5 0.5 0.95 0.95\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    first = detect.detect(str(config), str(image), method="hog_svm", labels=str(labels))
    labels.write_text("2 0.5 0.5 0.95 0.95\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    second = detect.detect(str(config), str(image), method="hog_svm", labels=str(labels))
    assert first["regions"] == second["regions"]
    assert first["person_count"] == 1
    assert len(first["excluded"]) == 1
    assert model_path.read_bytes() == frozen


def test_no_person_input_has_no_image_decision(dataset):
    root, _, config, _ = dataset
    train_model(config)
    labels = root / "empty.txt"
    labels.write_text("", encoding="utf-8")
    result = detect.detect(str(config), str(root / "test/images/empty.png"),
                           method="hog_svm", labels=str(labels))
    assert result["image_prediction"] is None
    assert result["person_count"] == 0
