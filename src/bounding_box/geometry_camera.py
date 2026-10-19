"""Camera person localisation followed by frozen Geometry posture rules."""

import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from bounding_box.geometry import load_config, load_saved_rules, predict_regions, regions_from_boxes


def person_class_ids(names, settings) -> list[int]:
    """Accept COCO people or the project's Fall/Up regions, excluding no-person."""
    names = dict(enumerate(names)) if isinstance(names, list) else names
    accepted = {"person", settings.fall_class.strip().casefold(), settings.upright_class.strip().casefold()}
    ids = sorted(int(i) for i, name in names.items() if str(name).strip().casefold() in accepted)
    if not ids:
        raise ValueError("Person detector must have a 'person' class or the configured Fall/Up classes")
    return ids


def annotate_frame(cv2, frame, prediction: dict) -> None:
    for region in prediction["regions"]:
        x1, y1, x2, y2 = (round(v) for v in region["bbox_xyxy"])
        color = (0, 0, 255) if region["prediction"] == "Fall" else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{region['prediction']} r={region['aspect_ratio']:.2f}",
                    (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status = prediction["image_prediction"] or "No person detected"
    cv2.putText(frame, f"Geometry: {status} | Q/Esc: quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def detect_camera(config_path: str, source: int, model_path: str | None = None,
                  rule_name: str | None = None, output: str | None = None,
                  detector_weights: str | None = None, detector_confidence: float = 0.25,
                  show: bool = True) -> dict:
    """Locate people per frame and apply saved rules, without fitting thresholds."""
    if not math.isfinite(detector_confidence) or not 0 <= detector_confidence <= 1:
        raise ValueError("Detector confidence must be between 0 and 1")
    settings = load_config(config_path)
    model_path = Path(model_path) if model_path is not None else Path(settings.output_dir) / "model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Geometry rules not found: {model_path}; run fall-train --method geometry first")
    model, rules = load_saved_rules(model_path, settings)
    chosen = rule_name or model["selected_rule_by_validation"]
    rule = next((r for r in rules if r.name == chosen), None)
    if rule is None:
        raise ValueError(f"Unknown geometry rule: {chosen}")
    detector_weights = detector_weights or "yolov8n.pt"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(output) if output is not None else (
        Path(settings.output_dir) / "predictions" / f"camera_{source}_{stamp}.jsonl")
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("Camera --output must be a .jsonl file (one result per frame)")
    if output_path.resolve() in {model_path.resolve(), Path(detector_weights).resolve(), Path(config_path).resolve()}:
        raise ValueError("Camera output must not overwrite saved rules, detector weights or configuration")

    # Still-image Geometry does not need either of these dependencies.
    import cv2
    from ultralytics import YOLO

    detector = YOLO(detector_weights)
    class_ids = person_class_ids(detector.names, settings)
    capture = cv2.VideoCapture(source)
    window = "Geometry fall detection"
    window_created = False
    frames = fall_frames = 0
    stopped_by = "camera_read_failed"
    started = time.perf_counter()
    last_prediction = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {source}; check the camera index and camera access")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_created = True
        print(f"Geometry camera {source}: detector={detector_weights}, rule={rule.name}")
        print("Press Q/Esc or close the window to stop; Ctrl+C also stops detection.")
        print(f"Frame results: {output_path}")
        with output_path.open("w", encoding="utf-8") as log:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if frames == 0:
                        raise RuntimeError(f"Camera {source} opened but no frame could be read")
                    print(f"Camera {source} stopped delivering frames.")
                    break
                height, width = frame.shape[:2]
                results = detector.predict(source=frame, classes=class_ids, conf=detector_confidence,
                                           verbose=False, show=False, save=False)
                boxes, confidences = [], []
                for result in results:
                    if result.boxes is None:
                        continue
                    # Convert GPU tensors only after localisation, retaining original-frame coordinates.
                    coordinates = result.boxes.xyxy.cpu().tolist()
                    classes = result.boxes.cls.cpu().tolist()
                    scores = result.boxes.conf.cpu().tolist()
                    for box, class_id, score in zip(coordinates, classes, scores):
                        if int(class_id) in class_ids:
                            boxes.append(box)
                            confidences.append(float(score))
                prediction = predict_regions(regions_from_boxes(width, height, boxes), rule)
                for region, score in zip(prediction["regions"], confidences):
                    region["detector_confidence"] = score
                frames += 1
                fall_frames += prediction["image_prediction"] == "Fall"
                last_prediction = prediction["image_prediction"]
                record = {"method": "geometry", "source": source, "frame_index": frames,
                          "elapsed_seconds": time.perf_counter() - started,
                          "image_width": width, "image_height": height,
                          "model_path": str(model_path), "rule": asdict(rule),
                          "input": "detected_person_boxes", "detector_weights": str(detector_weights),
                          **prediction}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                if show:
                    annotate_frame(cv2, frame, prediction)
                    cv2.imshow(window, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        stopped_by = "key"
                        break
                    try:
                        visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
                    except cv2.error:
                        visible = 0
                    if visible < 1:
                        stopped_by = "window_closed"
                        break
    except KeyboardInterrupt:
        stopped_by = "interrupted"
    finally:
        capture.release()
        if window_created:
            # Closing the window manually can already have destroyed it.
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return {"method": "geometry", "source": source, "frame_count": frames, "fall_frame_count": fall_frames,
            "last_prediction": last_prediction, "stopped_by": stopped_by, "output_path": str(output_path),
            "model_path": str(model_path), "rule": asdict(rule), "detector_weights": str(detector_weights)}
