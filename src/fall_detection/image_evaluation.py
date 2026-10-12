"""Image-level fall evaluation. Calibration reads validation data only.

This is separate from detection mAP and from supplied-box geometry scores.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import yaml

from .config import load_settings


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def class_names(value):
    names = dict(enumerate(value)) if isinstance(value, list) else {int(k): v for k, v in value.items()}
    if not names or len(set(names.values())) != len(names) or any(not isinstance(v, str) for v in names.values()):
        raise ValueError('Class names must be unique strings')
    return names


def read_split(data_yaml, split, fall_class='fall', upright_class='up', no_person_class='0 person'):
    """Support an explicit .../images directory; never silently resolve another split.

    Empty labels are negatives, missing labels are errors. Negative subgroup names
    describe annotations, not verified real-world content.
    """
    from .geometry import parse_yolo_box
    source = Path(data_yaml).resolve()
    data = yaml.safe_load(source.read_text(encoding='utf-8'))
    names = class_names(data['names'])
    if fall_class not in names.values():
        raise ValueError(f'Missing fall class: {fall_class}')
    root = Path(data.get('path') or '.')
    if not root.is_absolute():
        root = source.parent / root
    entry = data.get(split)
    if not isinstance(entry, str):
        raise ValueError(f'{split} must explicitly name an images directory')
    folder = Path(entry)
    folder = (folder if folder.is_absolute() else root / folder).resolve()
    if not folder.is_dir() or folder.name != 'images':
        raise ValueError(f'Expected existing .../images directory: {folder}')
    images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'} and p.is_file())
    if not images:
        raise ValueError(f'No images in {folder}')
    seen, rows = set(), []
    for image in images:
        relative = image.relative_to(folder)
        label = folder.parent / 'labels' / relative.with_suffix('.txt')
        if label in seen:
            raise ValueError(f'Ambiguous label ownership: {label}')
        seen.add(label)
        if not label.is_file():
            raise FileNotFoundError(f'Missing annotation: {label}; explicit empty files are required for background')
        present = set()
        for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            if line.strip():
                cid, *_ = parse_yolo_box(line, label, number, names)
                present.add(names[cid])
        positive = fall_class in present
        group = ('positive' if positive else 'up_only' if present == {upright_class}
                 else 'no_person_label' if present == {no_person_class}
                 else 'empty_annotation' if not present else 'other_negative')
        rows.append({'image': relative.as_posix(), 'source': str(image), 'true_fall': positive,
                     'negative_group': group, 'image_sha256': digest(image), 'label_sha256': digest(label)})
    return names, rows


def metrics(rows, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0, 1]')
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    groups = {}
    for row in rows:
        score = row['fall_score']
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Fall scores must be finite and in [0, 1]')
        predicted = score > threshold
        outcome = ('tp' if predicted else 'fn') if row['true_fall'] else ('fp' if predicted else 'tn')
        counts[outcome] += 1
        if not row['true_fall']:
            g = groups.setdefault(row['negative_group'], {'images': 0, 'false_positives': 0})
            g['images'] += 1
            g['false_positives'] += int(predicted)
    tp, fp, tn, fn = (counts[k] for k in ('tp', 'fp', 'tn', 'fn'))
    for g in groups.values():
        g['false_positive_rate'] = g['false_positives'] / g['images']
    return {**counts, 'sample_count': len(rows), 'fall_precision': tp/(tp+fp) if tp+fp else 0,
            'fall_recall': tp/(tp+fn) if tp+fn else 0, 'fall_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'negative_groups': groups, 'confusion_matrix': {'labels': ['No Fall', 'Fall'],
            'rows': 'true', 'columns': 'predicted', 'counts': [[tn, fp], [fn, tp]]}}


def select_threshold(rows, floor):
    if not 0 <= floor < 1:
        raise ValueError('Confidence floor must be in [0, 1)')
    if {r['true_fall'] for r in rows} != {True, False}:
        raise ValueError('Validation requires both positive and negative images')
    candidates = sorted({floor, 1.0, *(r['fall_score'] for r in rows if r['fall_score'] >= floor)})
    curve = [{'threshold': t, **{k: v for k, v in metrics(rows, t).items()
                              if k in ('tp', 'fp', 'tn', 'fn', 'fall_precision', 'fall_recall', 'fall_f1')}} for t in candidates]
    # Strict > implements the report's "exceeds" rule. Lowest threshold breaks remaining ties.
    best = max(curve, key=lambda r: (r['fall_f1'], r['fall_recall'], -r['threshold']))
    return best['threshold'], curve


def collect_scores(model, rows, names, protocol):
    model_names = class_names(model.names)
    if model_names != names:
        raise ValueError('Model class IDs/names do not match dataset; use trained fall weights')
    fall_id = next(k for k, v in names.items() if v == protocol['fall_class'])
    scored = []
    # One explicit warm-up pass, excluded from per-image timing.
    for index, row in enumerate([rows[0], *rows]):
        start = time.perf_counter()
        results = list(model.predict(source=row['source'], conf=protocol['confidence_floor'],
                       imgsz=protocol['image_size'], iou=protocol['nms_iou'], max_det=protocol['max_det'],
                       device=protocol['device'], rect=False, augment=False, agnostic_nms=False,
                       batch=1, verbose=False, save=False, show=False))
        if len(results) != 1:
            raise ValueError('Expected exactly one prediction result per image')
        boxes = results[0].boxes
        scores = [] if boxes is None else [float(s) for c, s in zip(boxes.cls.tolist(), boxes.conf.tolist()) if int(c) == fall_id]
        elapsed = (time.perf_counter() - start) * 1000
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
            raise ValueError('Fall scores must be finite and in [0, 1]')
        if index:
            scored.append({**row, 'fall_score': max(scores, default=0.0), 'processing_ms': elapsed})
    return scored


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def export_results(output, rows, threshold, metadata, curve=None):
    report = {**metadata, 'threshold': threshold, 'evaluation_unit': 'image', 'input': 'full image, no supplied boxes',
              'decision': 'highest fall score > threshold', **metrics(rows, threshold)}
    report['timing'] = {'mean_ms_per_image': sum(r['processing_ms'] for r in rows)/len(rows),
                        'warmup_images': 1, 'batch_size': 1,
                        'includes': 'image loading, preprocessing, inference, NMS, CPU score extraction',
                        'excludes': 'model loading, labels, hashing, warm-up, metrics and file export'}
    predictions = []
    for row in rows:
        predicted = row['fall_score'] > threshold
        predictions.append({**row, 'ground_truth': 'Fall' if row['true_fall'] else 'No Fall',
            'prediction': 'Fall' if predicted else 'No Fall',
            'outcome': ('TP' if predicted else 'FN') if row['true_fall'] else ('FP' if predicted else 'TN')})
    fields = ['image', 'source', 'ground_truth', 'prediction', 'fall_score', 'outcome', 'negative_group', 'processing_ms', 'image_sha256', 'label_sha256']
    write_csv(output / 'predictions.csv', predictions, fields)
    errors = [r for r in predictions if r['outcome'] in ('FP', 'FN')]
    write_csv(output / 'errors.csv', errors, fields)
    if curve is not None:
        write_csv(output / 'threshold_search.csv', curve, list(curve[0]))
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (output / 'summary.md').write_text(
        f"# YOLO image-level {metadata['stage']}\n\nImages: {len(rows)}; threshold: {threshold:.8g}.\n\n"
        f"Precision: {report['fall_precision']:.4f}; Recall: {report['fall_recall']:.4f}; F1: {report['fall_f1']:.4f}.\n\n"
        f"TP={report['tp']}, FP={report['fp']}, TN={report['tn']}, FN={report['fn']}.\n\n"
        "Negative groups reflect annotations; their semantic correctness has not been independently verified. "
        "These image-level scores must not be ranked directly against supplied-box person-level scores. "
        "Detection mAP is evaluated separately. Zero-denominator precision/recall/F1 use zero.\n", encoding='utf-8')
    lines = ['# Error review worksheet', '', 'No causes are inferred automatically. Inspect original images and labels.', '',
             '| Image | Outcome | Fall score | Observed cause / label issue |', '|---|---|---:|---|']
    lines += [f"| {r['image'].replace('|', '-')} | {r['outcome']} | {r['fall_score']:.4f} | Pending review |" for r in errors]
    (output / 'error_review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def run(config, weights, stage, output, calibration=None, model_factory=None):
    if stage not in {'calibrate', 'test'}:
        raise ValueError('Stage must be calibrate or test')
    settings = load_settings(config)
    weights = Path(weights).resolve()
    weight_hash = digest(weights)
    if stage == 'test':
        if calibration is None:
            raise ValueError('Testing requires a saved calibration.json')
        frozen = json.loads(Path(calibration).read_text(encoding='utf-8'))
        if frozen.get('schema_version') != 1 or frozen.get('selection_split') != 'val':
            raise ValueError('Invalid validation calibration artifact')
        if frozen['weights_sha256'] != weight_hash:
            raise ValueError('Weights differ from validation calibration')
        protocol = frozen['protocol']
        if settings.image_size != protocol['image_size']:
            raise ValueError('Image size differs from frozen calibration')
        threshold = frozen['threshold']
        if not protocol['confidence_floor'] <= threshold <= 1:
            raise ValueError('Invalid saved threshold')
    else:
        protocol = {'fall_class': 'fall', 'confidence_floor': 0.001, 'nms_iou': 0.7, 'max_det': 300,
                    'image_size': settings.image_size, 'device': None if settings.device == 'auto' else settings.device}
    names, rows = read_split(settings.data_yaml, 'val' if stage == 'calibrate' else 'test', protocol['fall_class'])
    if stage == 'test':
        if {str(k): v for k, v in names.items()} != frozen['class_names']:
            raise ValueError('Dataset classes differ from calibration')
        if {r['image_sha256'] for r in rows} & set(frozen['validation_image_hashes']):
            raise ValueError('Identical image content occurs in validation and test; repair the split')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory to preserve prior evidence: {output}')
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(weights))
    rows = collect_scores(model, rows, names, protocol)
    curve = None
    if stage == 'calibrate':
        threshold, curve = select_threshold(rows, protocol['confidence_floor'])
        frozen = {'schema_version': 1, 'selection_split': 'val', 'threshold': threshold,
                  'weights_sha256': weight_hash, 'weights_path': str(weights), 'protocol': protocol,
                  'class_names': {str(k): v for k, v in names.items()},
                  'validation_image_hashes': sorted({r['image_sha256'] for r in rows}),
                  'selection': 'maximum Fall F1, then recall, then lowest threshold'}
    versions = {}
    for package in ('ultralytics', 'torch', 'numpy', 'Pillow', 'PyYAML'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    metadata = {'stage': stage, 'split': 'val' if stage == 'calibrate' else 'test', 'weights_sha256': weight_hash,
                'weights_path': str(weights), 'protocol': protocol, 'dataset_yaml_sha256': digest(settings.data_yaml),
                'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'processor': platform.processor(), 'packages': versions},
                'calibration_sha256': digest(calibration) if calibration else None,
                'limitations': ['No semantic verification of 0 person or empty labels.',
                                'Near-duplicates, train/test leakage and shared source recordings require separate audit.',
                                'Scores are based on post-NMS detections above the recorded confidence floor.']}
    output.mkdir(parents=True, exist_ok=False)
    if stage == 'calibrate':
        (output / 'calibration.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    return export_results(output, rows, threshold, metadata, curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'test'])
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--weights', required=True, help='Trained fall detector, not generic pretrained weights')
    parser.add_argument('--output', required=True, help='New directory for reproducible results')
    parser.add_argument('--calibration', help='Frozen calibration.json required for test')
    args = parser.parse_args()
    result = run(args.config, args.weights, args.stage, args.output, args.calibration)
    print(f"Images={result['sample_count']}, Precision={result['fall_precision']:.4f}, "
          f"Recall={result['fall_recall']:.4f}, F1={result['fall_f1']:.4f}")


if __name__ == '__main__':
    main()
