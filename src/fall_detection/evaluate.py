import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import load_settings


def evaluate(config_path: str, weights: str | None = None) -> None:
    settings = load_settings(config_path)
    weights_path = weights or str(
        Path(settings.project_dir) / settings.run_name / "weights" / "best.pt"
    )
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(weights_path)
    metrics = model.val(data=settings.data_yaml)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fall detector")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    evaluate(args.config, args.weights)


if __name__ == "__main__":
    main()
