"""Display saved test evaluations without running training or inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load_reports(root):
  root = Path(root)
  specifications = [
    ('Bounding box', 'configs/geometry.yaml', 'output_dir', 'runs/geometry'),
    ('HOG-SVM', 'configs/hog_svm.yaml', 'output_dir', 'runs/hog_svm'),
    ('YOLOv8', 'configs/default.yaml', 'project_dir', 'runs/yolov8'),
  ]
  reports = []
  for name, config_path, key, default in specifications:
    entry = {'name': name}
    try:
      config = yaml.safe_load((root / config_path).read_text()) or {}
      output = Path(config.get(key, default))
      if not output.is_absolute():
        output = root / output
      path = output / 'evaluation/test/metrics.json'
      report = json.loads(path.read_text())
      if name == 'Bounding box':
        rule = report['model']['selected_rule_by_validation']
        scores = report['splits']['test']['rules'][rule]
        entry['detail'] = f'Supplied person boxes · {rule.replace("_", " ")}'
      else:
        if report['split'] != 'test':
          raise ValueError('Expected test-set metrics')
        scores = report
        entry[
          'detail'] = 'Supplied person boxes' if name == 'HOG-SVM' else 'Full-image detection'
      if name == 'YOLOv8':
        fall = report['per_class'].get('fall')
        entry['values'] = (
          [fall['precision'], fall['recall'], fall['f1']] if fall else [None] *
          3) + [report['map50'], report['map50_95']]
        entry['labels'] = ['Fall P', 'Fall R', 'Fall F1', 'mAP50', 'mAP50–95']
        entry['detail'] += '\nP/R/F1: fall class; mAP: all evaluated classes'
      else:
        entry['values'] = [
          scores[k] for k in ('fall_precision', 'fall_recall', 'fall_f1')
        ]
        entry['labels'] = ['Fall precision', 'Fall recall', 'Fall F1']
      entry['matrix'] = scores['confusion_matrix']
      matrix = np.asarray(entry['matrix']['counts'])
      n = len(entry['matrix']['labels'])
      if matrix.shape != (n,
                          n) or not np.isfinite(matrix).all() or (matrix
                                                                  < 0).any():
        raise ValueError('Invalid confusion matrix')
      if any(v is not None and (not np.isfinite(v) or not 0 <= v <= 1)
             for v in entry['values']):
        raise ValueError('Scores must be between zero and one')
      entry['path'] = str(path)
    except FileNotFoundError:
      entry = {
        'name':
        name,
        'error':
        f'No saved test evaluation yet.\nSelect method {len(reports)+1} → Evaluate,\nthen reopen this window.'
      }
    except (OSError, ValueError, KeyError, TypeError) as error:
      entry = {
        'name':
        name,
        'error':
        f'Cannot read evaluation: {error}\nRun this method’s test evaluation again.'
      }
    reports.append(entry)
  return reports


def build_figure(reports):
  import matplotlib.pyplot as plt
  from matplotlib.ticker import PercentFormatter

  fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
  fig.patch.set_facecolor('#f5f7fb')
  fig.suptitle(
    'Fall detection · test-set evaluations',
    fontsize=20,
    fontweight='bold',
    y=.97)
  colors = ['#237b86', '#6153a3', '#c07127']
  for i, report in enumerate(reports):
    top, bottom = axes[0, i], axes[1, i]
    top.set_title(report['name'], fontsize=15, fontweight='bold', pad=34)
    if 'error' in report:
      for ax in (top, bottom):
        ax.set_axis_off()
      top.text(
        .5,
        .5,
        report['error'],
        ha='center',
        va='center',
        transform=top.transAxes,
        fontsize=11,
        wrap=True)
      bottom.text(
        .5,
        .5,
        'Awaiting evaluation',
        ha='center',
        va='center',
        color='#7a8494',
        transform=bottom.transAxes)
      continue
    top.text(
      .5,
      1.025,
      report['detail'],
      ha='center',
      va='bottom',
      transform=top.transAxes,
      fontsize=9)
    for j, value in enumerate(report['values']):
      if value is None:
        top.text(j, .05, 'N/A', ha='center', fontsize=9)
      else:
        top.bar(j, value, color=colors[i], width=.62)
        top.text(j, value + .025, f'{value:.1%}', ha='center', fontsize=9)
    top.set_xticks(range(len(report['labels'])), report['labels'], fontsize=9)
    top.set_ylim(0, 1.12)
    top.set_yticks([0, .25, .5, .75, 1])
    top.yaxis.set_major_formatter(PercentFormatter(1))
    top.grid(axis='y', alpha=.2)
    top.set_axisbelow(True)
    top.spines[['top', 'right']].set_visible(False)
    matrix = np.asarray(report['matrix']['counts'])
    labels = report['matrix']['labels']
    bottom.imshow(matrix, cmap='Blues', vmin=0)
    for (r, c), value in np.ndenumerate(matrix):
      bottom.text(
        c,
        r,
        str(int(value)),
        ha='center',
        va='center',
        color='white' if value > matrix.max() / 2 else '#17314d',
        fontsize=11)
    bottom.set_xticks(range(len(labels)), labels, fontsize=9)
    bottom.set_yticks(range(len(labels)), labels, fontsize=9)
    bottom.set_xlabel('Predicted label')
    bottom.set_ylabel('True label')
    bottom.set_title('Confusion matrix · counts', fontsize=11, pad=10)
  fig.text(
    .5,
    .065,
    'Geometry and HOG-SVM classify supplied person regions; YOLOv8 also locates people. Scores are not a like-for-like ranking.',
    ha='center',
    fontsize=10)
  fig.text(
    .5,
    .035,
    'YOLOv8 P/R/F1 and its confusion matrix use different operating thresholds. Missing results are shown as pending, never zero.',
    ha='center',
    fontsize=9,
    color='#526077')
  fig.subplots_adjust(
    left=.065, right=.975, top=.84, bottom=.16, hspace=.52, wspace=.35)
  return fig


def display(root=None, show=True):
  import matplotlib.pyplot as plt
  root = Path(root) if root else Path(__file__).resolve().parents[2]
  reports = load_reports(root)
  fig = build_figure(reports)
  out = root / 'runs/comparison/test_evaluations.png'
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
  print(f'Evaluation charts saved to {out}')
  for report in reports:
    print(f'{report["name"]}: {report.get("path", report.get("error"))}')
  if show:
    fig.canvas.manager.set_window_title(
      'Fall detection — evaluation comparison')
    plt.show()
  plt.close(fig)
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    '--no-show', action='store_true', help='Save PNG without opening a window')
  args = parser.parse_args()
  display(show=not args.no_show)


if __name__ == '__main__':
  main()
