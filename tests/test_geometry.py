import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from bounding_box.geometry import (
    GeometryRule,
    GeometrySettings,
    Region,
    classification_metrics,
    load_split,
    run_experiment,
    tune_rules,
)


def region(ratio, position, fall, name="sample.jpg"):
    return Region(name, 1, "fall" if fall else "up", fall, 200, 100,
                  0.5, position, ratio * 0.1, 0.2)


def create_dataset(tmp_path, validation, test):
    root = tmp_path / "data"
    for split, labels in [("valid", validation), ("test", test)]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for i, label in enumerate(labels):
            Image.new("RGB", (200, 100)).save(root / split / "images" / f"{i}.png")
            (root / split / "labels" / f"{i}.txt").write_text(label, encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("val: valid/images\ntest: test/images\nnames: ['0 person', 'fall', 'up']\n", encoding="utf-8")
    config = tmp_path / "geometry.yaml"
    config.write_text(f"data_yaml: '{data_yaml.as_posix()}'\n"
                      f"output_dir: '{(tmp_path / 'results').as_posix()}'\ntiming_repeats: 2\n", encoding="utf-8")
    return root, data_yaml, config


def test_non_square_image_uses_pixel_ratio():
    sample = region(2.0, 0.5, True)
    assert sample.box_width / sample.box_height == 1.0
    assert sample.aspect_ratio == 2.0
    assert GeometryRule("aspect_ratio", 1.5).predict([sample]).tolist() == [True]


def test_rule_boundaries_are_strict_ratio_and_inclusive_position():
    rule = GeometryRule("aspect_ratio_position", 1.0, 0.7)
    assert rule.predict([region(1.0, 0.8, False), region(2.0, 0.7, True),
                         region(2.0, 0.6, True)]).tolist() == [False, True, False]


def test_position_can_separate_equal_shape_regions():
    samples = [region(2.0, 0.8, True), region(2.0, 0.3, False)]
    aspect, position = tune_rules(samples)[0]
    assert classification_metrics([True, False], aspect.predict(samples))["fall_f1"] == pytest.approx(2 / 3)
    assert position.predict(samples).tolist() == [True, False]


def test_threshold_ties_prioritise_recall():
    # Threshold 0: TP=2, FP=4, F1=.5. Threshold 1: TP=1, FP=1, F1=.5.
    samples = [region(2, 0.5, True), region(1, 0.5, True)]
    samples += [region(2, 0.5, False)] + [region(1, 0.5, False) for _ in range(3)]
    aspect = tune_rules(samples)[0][0]
    assert aspect.aspect_threshold == 0
    assert classification_metrics([r.true_fall for r in samples], aspect.predict(samples))["fall_recall"] == 1


def test_threshold_search_matches_independent_exhaustive_predictions():
    rng = np.random.default_rng(42)
    for _ in range(8):
        samples = [region(float(rng.choice([0.3, 0.7, 1.0, 1.7, 2.0])),
                          float(rng.choice([0.2, 0.5, 0.8])), i % 2 == 0)
                   for i in range(14)]
        rules, _ = tune_rules(samples)
        ratios = sorted({0.0, *(r.aspect_ratio for r in samples)})
        positions = sorted({0.0, 1.0, *(r.center_y for r in samples)})
        for rule in rules:
            options = []
            for rt, pt in itertools.product(ratios, [0.0] if rule.name == "aspect_ratio" else positions):
                predicted = [r.aspect_ratio > rt and r.center_y >= pt for r in samples]
                tp = sum(p and r.true_fall for p, r in zip(predicted, samples))
                fp = sum(p and not r.true_fall for p, r in zip(predicted, samples))
                positives = sum(r.true_fall for r in samples)
                options.append(((2 * tp / (positives + tp + fp), tp / positives, -pt, -rt), rt, pt))
            best = max(options)
            assert (rule.aspect_threshold, rule.position_threshold) == best[1:]


def test_no_person_and_empty_images_are_excluded_not_true_negatives(tmp_path):
    _, data_yaml, _ = create_dataset(tmp_path, ["0 0.5 0.5 0.4 0.4\n", "",
                                              "1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.1 0.3\n"], [""])
    split = load_split(data_yaml, "val", GeometrySettings())
    assert len(split.regions) == 2
    assert len(split.excluded) == 2
    assert split.summary()["evaluated_images"] == 1
    assert split.class_counts == {"0 person": 1, "fall": 1, "up": 1}


@pytest.mark.parametrize("bad_label", ["1 0.5 0.5 0.2 0", "1 nan 0.5 0.2 0.2",
                                       "7 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.2", "1 1.2 0.5 0.2 0.2"])
def test_invalid_annotations_fail_with_source_location(tmp_path, bad_label):
    _, data_yaml, _ = create_dataset(tmp_path, [bad_label], [""])
    with pytest.raises(ValueError, match=r"0.txt:1:"):
        load_split(data_yaml, "val", GeometrySettings())


def test_missing_annotation_is_not_silently_treated_as_no_fall(tmp_path):
    root, data_yaml, _ = create_dataset(tmp_path, [""], [""])
    (root / "valid/labels/0.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing annotation"):
        load_split(data_yaml, "val", GeometrySettings())


def test_metrics_confusion_and_undefined_no_person_rate():
    metrics = classification_metrics([True, True, False, False], [True, False, True, False])
    assert [metrics[k] for k in ["tp", "fp", "tn", "fn"]] == [1, 1, 1, 1]
    assert metrics["fall_f1"] == 0.5
    assert metrics["confusion_matrix"]["counts"] == [[1, 1], [1, 1]]
    assert metrics["no_person_false_positive_rate"] is None
    assert classification_metrics([], [])["up_false_positive_rate"] is None


def test_test_labels_cannot_change_validation_thresholds_and_saved_rules_are_reusable(tmp_path):
    root, _, config = create_dataset(tmp_path, ["1 0.5 0.8 0.2 0.2\n", "2 0.5 0.3 0.05 0.3\n"],
                                     ["1 0.5 0.8 0.2 0.2\n", "2 0.5 0.3 0.05 0.3\n"])
    first = run_experiment(config)
    (root / "test/labels/0.txt").write_text("2 0.5 0.8 0.2 0.2\n", encoding="utf-8")
    second = run_experiment(config)
    assert first["model"] == second["model"]
    assert first["splits"]["test"]["rules"]["aspect_ratio"]["tp"] == 1
    assert second["splits"]["test"]["rules"]["aspect_ratio"]["tp"] == 0
    model_path = tmp_path / "results/model.json"
    saved = json.loads(model_path.read_text())
    # Re-evaluation uses frozen thresholds even if validation labels have changed.
    (root / "valid/labels/0.txt").write_text("2 0.5 0.8 0.2 0.2\n", encoding="utf-8")
    reused = run_experiment(config, model_path)
    assert reused["model"] == saved
    predictions = (tmp_path / "results/test_aspect_ratio_predictions.csv").read_text()
    assert "region_id" in predictions and "original_class" in predictions
    assert (tmp_path / "results/test_aspect_ratio_errors.csv").is_file()


def test_model_is_frozen_before_test_inputs_are_read(tmp_path):
    root, _, config = create_dataset(tmp_path, ["1 0.5 0.8 0.2 0.2\n", "2 0.5 0.3 0.05 0.3\n"], [""])
    (root / "test/labels/0.txt").unlink()
    with pytest.raises(FileNotFoundError):
        run_experiment(config)
    assert (tmp_path / "results/model.json").is_file()


def test_tuning_requires_both_person_classes():
    with pytest.raises(ValueError, match="both fall and up"):
        tune_rules([region(1, 0.5, True)])


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_timing_configuration(value):
    with pytest.raises(ValueError, match="timing_repeats"):
        replace(GeometrySettings(), timing_repeats=value).validate()
