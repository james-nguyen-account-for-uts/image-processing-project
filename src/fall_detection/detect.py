import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import load_settings


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def detect(config_path: str, source: str, weights: str | None = None) -> None:
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
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    detect(args.config, args.source, args.weights)


if __name__ == "__main__":
    main()
