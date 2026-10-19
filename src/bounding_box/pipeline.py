"""Public bounding-box method API."""
from pathlib import Path
from .geometry import calibrate, evaluate_saved, print_results, load_config, load_saved_rules, regions_from_boxes, predict_regions


def train(config_path='configs/geometry.yaml'):
    result = calibrate(config_path)
    print_results(result)
    return result


def evaluate(config_path='configs/geometry.yaml', split='test'):
    result = evaluate_saved(config_path, split=split)
    print_results(result)
    return result


def detect(config_path='configs/geometry.yaml', source='0', labels=None, show=True):
    config = load_config(config_path)
    model_path = Path(config.output_dir)/'model.json'
    if not model_path.is_file():
        raise FileNotFoundError(f'Train the bounding-box method first: missing {model_path}')
    model, rules = load_saved_rules(model_path, config)
    rule = next(r for r in rules if r.name == model['selected_rule_by_validation'])
    def classify(image, boxes):
        return predict_regions(regions_from_boxes(*image.size, boxes),rule)['regions']
    from common.inference import run
    return run(source, classify, config.output_dir, 'Bounding box', labels=labels,
               data_yaml=config.data_yaml, show=show)
