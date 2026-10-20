"""HOG + linear SVM on annotated person crops; no test data used for fitting."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from PIL import Image
import yaml
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common.dataset import load_split, classification_metrics


def settings(path='configs/hog_svm.yaml'):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('HOG configuration must be a mapping')
    config.setdefault('fall_class', 'fall')
    config.setdefault('upright_class', 'up')
    config.setdefault('no_person_class', '0 person')
    if not config.get('c_values') or any(float(c) <= 0 for c in config['c_values']):
        raise ValueError('c_values must contain positive SVM regularization values')
    return SimpleNamespace(**config)


def features(image):
    """Fixed geometry is saved with the model; training and inference share this path."""
    image = image.convert('L').resize((64, 128), Image.Resampling.BILINEAR)
    return hog(np.asarray(image, dtype=np.float32) / 255.0, orientations=9,
               pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
               transform_sqrt=True).astype(np.float32)


def crop(image, box):
    width, height = image.size
    x1, y1, x2, y2 = map(float, box)
    if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
        raise ValueError('Expected a finite positive-area person box')
    bounds = (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
              min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError('Person box does not intersect the image')
    return image.crop(bounds)


def dataset(config, split):
    records = load_split(config.data_yaml, split, config)
    if not records.regions:
        raise ValueError(f'No person regions in {split}')
    yaml_path = Path(config.data_yaml).resolve()
    metadata = yaml.safe_load(yaml_path.read_text())
    root = Path(metadata.get('path') or '.')
    if not root.is_absolute():
        root = yaml_path.parent / root
    x, y = [], []
    for r in records.regions:
        box = ((r.center_x-r.box_width/2)*r.image_width,
               (r.center_y-r.box_height/2)*r.image_height,
               (r.center_x+r.box_width/2)*r.image_width,
               (r.center_y+r.box_height/2)*r.image_height)
        with Image.open(root / r.image) as image:
            x.append(features(crop(image, box)))
        y.append(r.true_fall)
    return np.asarray(x), np.asarray(y), records


def train(config_path='configs/hog_svm.yaml'):
    config = settings(config_path)
    print('Extracting HOG features from train and validation person crops...', flush=True)
    x, y, training = dataset(config, 'train')
    vx, vy, validation = dataset(config, 'val')
    if len(np.unique(y)) != 2 or len(np.unique(vy)) != 2:
        raise ValueError('Training and validation both need fall and up regions')
    candidates = []
    best, best_score = None, None
    for c in config.c_values:
        pipeline = make_pipeline(StandardScaler(), LinearSVC(C=float(c), class_weight='balanced',
                                                            dual=False, max_iter=20000, random_state=42))
        pipeline.fit(x, y)
        metrics = classification_metrics(vy, pipeline.predict(vx))
        candidates.append({'C': c, **metrics})
        score = (metrics['fall_f1'], metrics['fall_recall'], -float(c))
        if best_score is None or score > best_score:
            best, best_score = (pipeline, c), score
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'schema_version': 1, 'feature_spec': 'gray64x128-hog9-cell8-block2-l2hys-sqrt',
                 'classifier': best[0], 'C': best[1], 'classes': ['up', 'fall']}, out/'model.joblib')
    report = {'method': 'hog_svm', 'fit_split': 'train', 'selection_split': 'val',
              'selected_C': best[1], 'train': training.summary(), 'val': validation.summary(),
              'candidates': candidates, 'evaluation_unit': 'annotated_person_region'}
    (out/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'HOG–SVM saved to {out / "model.joblib"}; validation-selected C={best[1]}')
    return report


def load_model(config):
    path = Path(config.output_dir)/'model.joblib'
    if not path.is_file():
        raise FileNotFoundError(f'Train HOG–SVM first: missing {path}')
    model = joblib.load(path)
    if model.get('schema_version') != 1 or model.get('feature_spec') != 'gray64x128-hog9-cell8-block2-l2hys-sqrt':
        raise ValueError('Unsupported HOG model format; retrain the classifier')
    return model['classifier']


def evaluate(config_path='configs/hog_svm.yaml', split='test'):
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    config = settings(config_path)
    model = load_model(config)
    x, y, records = dataset(config, split)
    predictions = model.predict(x)
    report = {'method': 'hog_svm', 'split': split, 'evaluation_unit': 'annotated_person_region',
              'dataset': records.summary(), **classification_metrics(y, predictions),
              'limitations': ['Supplied person boxes exclude localisation errors.',
                              'No-person regions are excluded, not counted as true negatives.']}
    out = Path(config.output_dir)/'evaluation'/split
    out.mkdir(parents=True, exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    import csv
    with (out/'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'region_id', 'true_fall', 'predicted_fall'])
        writer.writerows((r.image, r.region_id, bool(t), bool(p))
                         for r, t, p in zip(records.regions, y, predictions))
    print(f'{split}: precision={report["fall_precision"]:.4f}, recall={report["fall_recall"]:.4f}, F1={report["fall_f1"]:.4f}')
    print(f'Results: {out}')
    return report


def predict_boxes(image, boxes, model):
    if not boxes:
        return []
    x = np.asarray([features(crop(image, box)) for box in boxes])
    return [{'bbox_xyxy': list(map(float, box)), 'prediction': 'Fall' if p else 'No Fall'}
            for box, p in zip(boxes, model.predict(x))]


def detect(config_path='configs/hog_svm.yaml', source='0', labels=None, show=True):
    config = settings(config_path)
    model = load_model(config)
    from common.inference import run
    return run(source, lambda image, boxes: predict_boxes(image, boxes, model),
               config.output_dir, 'HOG-SVM', labels=labels, data_yaml=config.data_yaml, show=show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'evaluate', 'detect'])
    parser.add_argument('--config', default='configs/hog_svm.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--source', default='0')
    parser.add_argument('--labels', help='Optional YOLO person annotations for a still image')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.action == 'train':
        train(args.config)
    elif args.action == 'evaluate':
        evaluate(args.config, args.split)
    else:
        detect(args.config, args.source, args.labels, not args.no_show)


if __name__ == '__main__':
    main()
