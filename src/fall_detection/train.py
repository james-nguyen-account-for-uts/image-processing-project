import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import load_settings


def train(config_path: str) -> None:
    settings = load_settings(config_path)
    if not Path(settings.data_yaml).is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {settings.data_yaml}. See data/README.md."
        )

    model = YOLO(settings.model)
    model.train(
        data=settings.data_yaml,
        epochs=settings.epochs,
        imgsz=settings.image_size,
        batch=settings.batch_size,
        device=None if settings.device == "auto" else settings.device,
        project=settings.project_dir,
        name=settings.run_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fall detector")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
