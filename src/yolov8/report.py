"""Persist evaluation metrics for the comparison window."""
from datetime import datetime, timezone
import json
from pathlib import Path


def save_report(metrics, project_dir, split, weights, data_yaml):
    names = metrics.names
    names = dict(enumerate(names)) if isinstance(names, list) else names
    names = {int(k): str(v) for k, v in names.items()}
    per_class = {}
    for i, class_id in enumerate(metrics.box.ap_class_index):
        per_class[names[int(class_id)]] = {
            'precision': float(metrics.box.p[i]), 'recall': float(metrics.box.r[i]),
            'f1': float(metrics.box.f1[i]),
        }
    # Ultralytics stores predicted rows / true columns; transpose to match the other methods.
    matrix = metrics.confusion_matrix.matrix.T.astype(int).tolist()
    report = {
        'method': 'yolov8', 'split': split, 'evaluation_unit': 'full_image_detection',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'weights': str(Path(weights).resolve()), 'data_yaml': str(Path(data_yaml).resolve()),
        'per_class': per_class, 'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
        'confusion_matrix': {'labels': [names[i] for i in range(len(names))] + ['background'],
                             'rows': 'true', 'columns': 'predicted', 'counts': matrix},
        'notes': 'Class P/R/F1 use the Ultralytics curve operating point. The confusion matrix uses its own detection threshold.',
    }
    out = Path(project_dir)/'evaluation'/split/'metrics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(out)
    print(f'Comparison metrics saved to {out}')
    return out
