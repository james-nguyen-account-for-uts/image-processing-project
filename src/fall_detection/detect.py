import argparse
from pathlib import Path

from .config import add_method_arguments, load_settings, resolve_config


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def detect(config_path: str | None, source: str, weights: str | None = None, *,
           method: str = "yolo", boxes: list[list[float]] | None = None,
           labels: str | None = None, rule: str | None = None, output: str | None = None,
           detector_weights: str | None = None, detector_confidence: float | None = None,
           no_show: bool = False):
    config_path = resolve_config(method, config_path)
    if method == "geometry":
        if isinstance(parse_source(source), int):
            if boxes is not None or labels is not None:
                raise ValueError("Geometry camera mode locates people automatically; do not supply --bbox or --labels")
            from .geometry_camera import detect_camera

            result = detect_camera(config_path, parse_source(source), weights, rule, output, detector_weights,
                                   0.25 if detector_confidence is None else detector_confidence, show=not no_show)
            print(f"Camera stopped: {result['stopped_by']}; frames={result['frame_count']}")
            print(f"Results saved to {result['output_path']}")
            return result
        if detector_weights is not None or detector_confidence is not None or no_show:
            raise ValueError("--detector-weights, --detector-confidence and --no-show require a Geometry camera source")
        from .geometry import predict_image

        result = predict_image(config_path, source, weights, boxes, labels, rule, output)
        for region in result["regions"]:
            print(f"Region {region['region_id']}: {region['prediction']}, aspect_ratio={region['aspect_ratio']:.4f}")
        print(f"Image: {result['image_prediction'] or 'No person region supplied'}")
        print(f"Result saved to {result['output_path']}")
        return result
    if any(value is not None for value in (boxes, labels, rule, output, detector_weights, detector_confidence)) or no_show:
        raise ValueError("Geometry detection options require --method geometry")

    from ultralytics import YOLO

    settings = load_settings(config_path)
    weights_path = weights or str(
        Path(settings.project_dir) / settings.run_name / "weights" / "best.pt"
    )
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(weights_path)
    model.predict(
        source=parse_source(source),
        conf=settings.confidence,
        show=True,
        save=True,
        project=settings.project_dir,
        name="predictions",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fall detection")
    add_method_arguments(parser)
    parser.add_argument("--source", required=True, help="Image, video, stream or camera index; Geometry supports images and camera indices")
    parser.add_argument("--weights", "--model", dest="weights", help="YOLO .pt weights or Geometry model.json")
    boxes = parser.add_mutually_exclusive_group()
    boxes.add_argument("--bbox", type=float, nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                       help="Geometry person box in pixels; repeat for multiple people")
    boxes.add_argument("--labels", help="YOLO annotation file supplying person boxes for Geometry")
    parser.add_argument("--rule", choices=["aspect_ratio", "aspect_ratio_position"],
                        help="Geometry rule override; defaults to the validation-selected rule")
    parser.add_argument("--output", help="Geometry result path: .json for images, .jsonl for cameras")
    parser.add_argument("--detector-weights", help="Geometry camera person detector (default: yolo11n.pt); separate from --weights model.json")
    parser.add_argument("--detector-confidence", type=float, help="Geometry camera person detection confidence (default: 0.25)")
    parser.add_argument("--no-show", action="store_true", help="Geometry camera: disable display; stop with Ctrl+C")
    args = parser.parse_args()
    camera = isinstance(parse_source(args.source), int)
    if args.method == "geometry":
        if camera and (args.bbox is not None or args.labels is not None):
            parser.error("Geometry camera mode locates people automatically; do not supply --bbox or --labels")
        if not camera and args.bbox is None and args.labels is None:
            parser.error("Geometry image detection requires --bbox or --labels; use --source 0 for a camera")
        if not camera and (args.detector_weights is not None or args.detector_confidence is not None or args.no_show):
            parser.error("--detector-weights, --detector-confidence and --no-show require a Geometry camera source")
        if args.detector_confidence is not None and not 0 <= args.detector_confidence <= 1:
            parser.error("--detector-confidence must be between 0 and 1")
    elif any(value is not None for value in (args.bbox, args.labels, args.rule, args.output,
                                           args.detector_weights, args.detector_confidence)) or args.no_show:
        parser.error("Geometry detection options require --method geometry")
    detect(args.config, args.source, args.weights, method=args.method, boxes=args.bbox,
           labels=args.labels, rule=args.rule, output=args.output, detector_weights=args.detector_weights,
           detector_confidence=args.detector_confidence, no_show=args.no_show)


if __name__ == "__main__":
    main()
