from dataclasses import dataclass
from pathlib import Path

import yaml


def resolve_config(method: str, path: str | None) -> str:
    if method not in {"yolo", "geometry"}:
        raise ValueError(f"Unknown detection method: {method}")
    return path or ("configs/geometry.yaml" if method == "geometry" else "configs/default.yaml")


def add_method_arguments(parser) -> None:
    parser.add_argument("--method", choices=["yolo", "geometry"], default="yolo")
    parser.add_argument("--config", help="Defaults to configs/default.yaml or configs/geometry.yaml by method")


@dataclass(frozen=True)
class Settings:
    model: str = "yolov8n.pt"
    data_yaml: str = "data/data.yaml"
    epochs: int = 30
    image_size: int = 640
    batch_size: int = 16
    device: str = "auto"
    confidence: float = 0.25
    project_dir: str = "runs/yolov8"
    run_name: str = "train"

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}

    settings = Settings(**values)
    settings.validate()
    return settings
