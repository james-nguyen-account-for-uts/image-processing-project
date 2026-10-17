import argparse
from pathlib import Path

from .config import add_method_arguments, load_settings, resolve_config


def evaluate(config_path: str | None = None, weights: str | None = None, *,
             method: str = "yolo", split: str | None = None):
    config_path = resolve_config(method, config_path)
    if split not in {None, "val", "test"}:
        raise ValueError("split must be val or test")
    if method == "geometry":
        from .geometry import evaluate_saved, print_results

        results = evaluate_saved(config_path, weights, split or "test")
        print_results(results)
        return results
    if method == "hog_svm":
        from .hog_svm import evaluate_model

        results = evaluate_model(config_path, weights, split or "test")
        m = results["metrics"]
        print(f"{results['split']} HOG-SVM: regions={m['sample_count']}, precision={m['fall_precision']:.4f}, "
              f"recall={m['fall_recall']:.4f}, F1={m['fall_f1']:.4f}")
        print(f"Results saved to {results['output_dir']}")
        return results

    from ultralytics import YOLO

    settings = load_settings(config_path)
    weights_path = weights or str(
        Path(settings.project_dir) / settings.run_name / "weights" / "best.pt"
    )
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(weights_path)
    options = {"data": settings.data_yaml}
    if split is not None:
        options["split"] = split
    metrics = model.val(**options)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fall detector")
    add_method_arguments(parser)
    parser.add_argument("--weights", "--model", dest="weights", help="YOLO .pt, Geometry .json, or HOG-SVM .joblib")
    parser.add_argument("--split", choices=["val", "test"], help="Defaults to val for YOLO; test for Geometry/HOG-SVM")
    args = parser.parse_args()
    evaluate(args.config, args.weights, method=args.method, split=args.split)


if __name__ == "__main__":
    main()
