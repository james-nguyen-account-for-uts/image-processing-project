import json
import sys
import types

import numpy as np
import pytest

from fall_detection import detect, geometry, geometry_camera


@pytest.fixture
def saved_camera_rules(tmp_path):
    config = tmp_path / "geometry.yaml"
    config.write_text(f"output_dir: '{tmp_path.as_posix()}'\n", encoding="utf-8")
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "schema_version": 1, "evaluation_unit": "annotated_person_region",
        "class_mapping": {"fall": "fall", "up": "up", "excluded_no_person": "0 person"},
        "rules": [{"name": "aspect_ratio", "aspect_threshold": 1.0, "position_threshold": 0},
                  {"name": "aspect_ratio_position", "aspect_threshold": 1.0, "position_threshold": 0.6}],
        "selected_rule_by_validation": "aspect_ratio",
    }), encoding="utf-8")
    return config, model


@pytest.fixture
def camera_runtime(monkeypatch):
    """Exercise the real camera loop with deterministic frames and GPU-like boxes."""
    runtime = types.SimpleNamespace(
        opened=True, frames=[np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)],
        detections=[([[10, 20, 170, 60], [10, 5, 40, 95], [0, 0, 190, 90]], [0, 0, 7], [0.9, 0.8, 0.99]),
                    ([], [], []), ([[10, 5, 40, 95]], [0], [0.8])],
        names={0: "person", 7: "car"}, calls=[], released=False, destroyed=False,
        displayed=0, keys=[], window_visible=1, predict_error=None, read_error=None)

    class Tensor:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeYOLO:
        def __init__(self, weights):
            runtime.calls.append(("load", weights))
            self.names = runtime.names

        def predict(self, **kwargs):
            runtime.calls.append(("predict", kwargs))
            if runtime.predict_error:
                raise runtime.predict_error
            xyxy, classes, confidence = runtime.detections.pop(0)
            return [types.SimpleNamespace(boxes=types.SimpleNamespace(
                xyxy=Tensor(xyxy), cls=Tensor(classes), conf=Tensor(confidence)))]

    class Capture:
        def __init__(self, source):
            runtime.source = source

        def isOpened(self):
            return runtime.opened

        def read(self):
            if runtime.read_error:
                raise runtime.read_error
            return (True, runtime.frames.pop(0)) if runtime.frames else (False, None)

        def release(self):
            runtime.released = True

    def imshow(*args):
        runtime.displayed += 1

    def destroy(*args):
        runtime.destroyed = True

    cv2 = types.SimpleNamespace(
        VideoCapture=Capture, WINDOW_NORMAL=0, FONT_HERSHEY_SIMPLEX=0, WND_PROP_VISIBLE=4,
        error=RuntimeError, namedWindow=lambda *args: None, rectangle=lambda *args: None,
        putText=lambda *args: None, imshow=imshow, destroyWindow=destroy,
        waitKey=lambda *args: runtime.keys.pop(0) if runtime.keys else -1,
        getWindowProperty=lambda *args: runtime.window_visible)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return runtime


def read_frames(summary):
    from pathlib import Path
    return [json.loads(line) for line in Path(summary["output_path"]).read_text(encoding="utf-8").splitlines()]


def test_camera_uses_frozen_rules_original_frame_pixels_and_no_person_state(saved_camera_rules, camera_runtime, monkeypatch):
    config, model = saved_camera_rules
    original = model.read_bytes()

    def no_training(*args):
        pytest.fail("Camera detection must never load a dataset or recalibrate")

    monkeypatch.setattr(geometry, "load_split", no_training)
    monkeypatch.setattr(geometry, "tune_rules", no_training)
    summary = detect.detect(str(config), "0", method="geometry")
    records = read_frames(summary)
    assert [r["image_prediction"] for r in records] == ["Fall", None, "No Fall"]
    assert [r["person_count"] for r in records] == [2, 0, 1]
    assert records[0]["regions"][0]["aspect_ratio"] == pytest.approx(4.0)
    assert records[0]["regions"][0]["normalized_center_y"] == pytest.approx(0.4)
    assert records[0]["regions"][0]["detector_confidence"] == 0.9
    assert records[0]["regions"][1]["prediction"] == "No Fall"
    assert summary["frame_count"] == 3 and summary["fall_frame_count"] == 1
    assert model.read_bytes() == original
    assert camera_runtime.source == 0
    assert camera_runtime.released and camera_runtime.destroyed and camera_runtime.displayed == 3
    assert camera_runtime.calls[0] == ("load", "yolo11n.pt")
    for kind, call in camera_runtime.calls[1:]:
        assert kind == "predict"
        assert call["classes"] == [0] and call["conf"] == 0.25
        assert call["show"] is False and call["save"] is False


def test_camera_cli_routes_detector_options_and_position_rule(saved_camera_rules, camera_runtime, monkeypatch):
    config, model = saved_camera_rules
    result_path = config.parent / "custom.jsonl"
    monkeypatch.setattr(sys, "argv", ["fall-detect", "--method", "geometry", "--config", str(config),
                                    "--source", "2", "--weights", str(model),
                                    "--detector-weights", "people.pt", "--detector-confidence", "0.7",
                                    "--rule", "aspect_ratio_position", "--no-show", "--output", str(result_path)])
    detect.main()
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["image_prediction"] == "No Fall"  # ratio passes, low-position test fails
    assert camera_runtime.source == 2 and camera_runtime.calls[0] == ("load", "people.pt")
    assert camera_runtime.calls[1][1]["conf"] == 0.7
    assert camera_runtime.released and camera_runtime.displayed == 0 and not camera_runtime.destroyed


def test_project_detector_class_predictions_do_not_decide_falls(saved_camera_rules, camera_runtime):
    config, _ = saved_camera_rules
    camera_runtime.names = {0: "0 person", 1: "fall", 2: "up"}
    # An Up detection with a wide box becomes Fall; a Fall detection with a tall box becomes No Fall.
    camera_runtime.detections[0] = ([[10, 20, 170, 60], [10, 5, 40, 95]], [2, 1], [0.9, 0.9])
    camera_runtime.keys = [ord("q")]
    summary = detect.detect(str(config), "0", method="geometry")
    assert [r["prediction"] for r in read_frames(summary)[0]["regions"]] == ["Fall", "No Fall"]
    assert camera_runtime.calls[1][1]["classes"] == [1, 2]


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_camera_exit_keys_release_resources(saved_camera_rules, camera_runtime, key):
    config, _ = saved_camera_rules
    camera_runtime.keys = [key]
    summary = detect.detect(str(config), "0", method="geometry")
    assert summary["stopped_by"] == "key" and summary["frame_count"] == 1
    assert camera_runtime.released and camera_runtime.destroyed


def test_closing_camera_window_releases_resources(saved_camera_rules, camera_runtime):
    config, _ = saved_camera_rules
    camera_runtime.window_visible = 0
    summary = detect.detect(str(config), "0", method="geometry")
    assert summary["stopped_by"] == "window_closed" and summary["frame_count"] == 1
    assert camera_runtime.released and camera_runtime.destroyed


def test_destroyed_camera_window_is_a_normal_exit(saved_camera_rules, camera_runtime, monkeypatch):
    config, _ = saved_camera_rules

    def destroyed_window(*args):
        raise RuntimeError("Window has already been destroyed")

    monkeypatch.setattr(sys.modules["cv2"], "getWindowProperty", destroyed_window)
    monkeypatch.setattr(sys.modules["cv2"], "destroyWindow", destroyed_window)
    summary = detect.detect(str(config), "0", method="geometry")
    assert summary["stopped_by"] == "window_closed"
    assert camera_runtime.released


def test_camera_keyboard_interrupt_releases_resources(saved_camera_rules, camera_runtime):
    config, _ = saved_camera_rules
    camera_runtime.read_error = KeyboardInterrupt()
    summary = detect.detect(str(config), "0", method="geometry", no_show=True)
    assert summary["stopped_by"] == "interrupted"
    assert camera_runtime.released


@pytest.mark.parametrize("failure", ["open", "read", "predict"])
def test_camera_failures_release_resources(saved_camera_rules, camera_runtime, failure):
    config, _ = saved_camera_rules
    if failure == "open":
        camera_runtime.opened = False
    elif failure == "read":
        camera_runtime.frames = []
    else:
        camera_runtime.predict_error = RuntimeError("Detector failed")
    with pytest.raises(RuntimeError):
        detect.detect(str(config), "0", method="geometry")
    assert camera_runtime.released
    assert camera_runtime.destroyed is (failure != "open")


def test_detector_without_person_classes_is_rejected_before_camera_open(saved_camera_rules, camera_runtime):
    config, _ = saved_camera_rules
    camera_runtime.names = ["cat", "car"]
    with pytest.raises(ValueError, match="Person detector"):
        detect.detect(str(config), "0", method="geometry")
    assert not hasattr(camera_runtime, "source")


@pytest.mark.parametrize("extra", [["--bbox", "0", "0", "10", "10"], ["--labels", "frame.txt"],
                                   ["--detector-confidence", "nan"], ["--detector-confidence", "1.1"]])
def test_camera_cli_invalid_inputs(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["fall-detect", "--method", "geometry", "--source", "0", *extra])
    with pytest.raises(SystemExit) as error:
        detect.main()
    assert error.value.code == 2


@pytest.mark.parametrize("argv", [
    ["--source", "0", "--detector-weights", "people.pt"],
    ["--source", "0", "--no-show"],
    ["--method", "geometry", "--source", "photo.jpg", "--bbox", "0", "0", "10", "10", "--no-show"],
])
def test_camera_options_do_not_leak_into_other_modes(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["fall-detect", *argv])
    with pytest.raises(SystemExit) as error:
        detect.main()
    assert error.value.code == 2


def test_camera_requires_saved_rules_and_jsonl_output(saved_camera_rules, camera_runtime):
    config, model = saved_camera_rules
    with pytest.raises(ValueError, match=".jsonl"):
        detect.detect(str(config), "0", method="geometry", output=str(config.parent / "result.json"))
    model.unlink()
    with pytest.raises(FileNotFoundError, match="fall-train --method geometry"):
        detect.detect(str(config), "0", method="geometry")
    assert not hasattr(camera_runtime, "source")
