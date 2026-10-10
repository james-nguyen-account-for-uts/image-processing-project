import argparse
from pathlib import Path

from .config import add_method_arguments, load_settings, resolve_config


def train(config_path: str | None = None, *, method: str = "yolo"):
    config_path = resolve_config(method, config_path)
    if method == "geometry":
        from .geometry import calibrate, print_results

        results = calibrate(config_path)
        print_results(results)
        print(f"Geometry rules saved to {results['model_path']}")
        return results

    from ultralytics import YOLO

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
    add_method_arguments(parser)
    args = parser.parse_args()
    train(args.config, method=args.method)


if __name__ == "__main__":
    main()
