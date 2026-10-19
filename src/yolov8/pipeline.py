"""Public YOLOv8 training, evaluation and inference API."""
from .train import train as _train
from .evaluate import evaluate as _evaluate
from .detect import detect as _detect


def train(config_path='configs/default.yaml'):
    return _train(config_path)


def evaluate(config_path='configs/default.yaml', split='test'):
    return _evaluate(config_path, split=split)


def detect(config_path='configs/default.yaml', source='0', show=True):
    return _detect(config_path, source, no_show=not show)
