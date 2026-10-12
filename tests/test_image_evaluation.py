import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from fall_detection.image_evaluation import metrics, select_threshold, read_split, run


def sample(truth, score, group='up_only'):
    return {'true_fall': truth, 'fall_score': score, 'negative_group': 'positive' if truth else group}


def test_image_metrics_missed_fall_and_negative_denominators():
    result = metrics([sample(True, 0), sample(True, .8), sample(False, .7),
                      sample(False, 0), sample(False, .9, 'no_person_label')], .5)
    assert (result['tp'], result['fp'], result['tn'], result['fn']) == (1, 2, 1, 1)
    assert result['fall_precision'] == pytest.approx(1/3)
    assert result['fall_recall'] == .5
    assert result['fall_f1'] == .4
    assert result['negative_groups']['up_only']['false_positive_rate'] == .5
    assert result['negative_groups']['no_person_label']['false_positive_rate'] == 1


def test_threshold_search_strict_boundary_and_no_detection():
    rows = [sample(True, .8), sample(True, .4), sample(False, .4), sample(False, 0)]
    threshold, _ = select_threshold(rows, .001)
    assert threshold == .001  # F1 .8 versus 2/3 after excluding the tied .4 scores
    assert metrics([sample(True, .4), sample(False, 0)], .4)['fn'] == 1
    assert metrics([sample(False, 0)], 0)['tn'] == 1
    with pytest.raises(ValueError, match='both'):
        select_threshold([sample(True, .8)], .001)


def test_equal_f1_prefers_higher_recall():
    rows = [sample(True, .9), sample(True, .4), sample(False, .6), sample(False, .5)]
    chosen, _ = select_threshold(rows, .001)
    assert metrics(rows, chosen)['fall_f1'] == pytest.approx(metrics(rows, .6)['fall_f1'])
    assert metrics(rows, chosen)['fall_recall'] == 1


@pytest.fixture
def project(tmp_path):
    # Non-standard class order ensures fall is resolved by name, not fixed ID.
    names = ['up', '0 person', 'fall']
    for split, offset in [('valid', 0), ('test', 40)]:
        images = tmp_path / split / 'images'
        labels = tmp_path / split / 'labels'
        images.mkdir(parents=True); labels.mkdir()
        for i, (name, text) in enumerate([('fall', '2 .5 .5 .8 .3\n0 .5 .5 .2 .8'),
                                         ('up', '0 .5 .5 .2 .8'), ('empty', ''),
                                         ('none', '1 .5 .5 .3 .3')]):
            Image.new('RGB', (20, 10), (offset+i, 0, 0)).save(images / f'{name}.png')
            (labels / f'{name}.txt').write_text(text)
    data = tmp_path / 'data.yaml'
    data.write_text(yaml.safe_dump({'names': names, 'val': 'valid/images', 'test': 'test/images'}))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'data_yaml': str(data)}))
    weights = tmp_path / 'trained.pt'; weights.write_bytes(b'fake trained model for tests only')
    calls = []

    class FakeModel:
        def __init__(self, path):
            self.names = dict(enumerate(names))

        def predict(self, **kwargs):
            calls.append(kwargs)
            name = Path(kwargs['source']).stem
            score = {'fall': .8, 'up': .3, 'none': .2, 'empty': 0}[name]
            # A very confident upright detection must not set the fall score.
            return [SimpleNamespace(boxes=SimpleNamespace(cls=np.array([0, 2]), conf=np.array([.99, score])))]

    return tmp_path, config, data, weights, FakeModel, calls


def test_calibrate_without_test_and_export_then_frozen_test(project):
    root, config, data, weights, model, calls = project
    # Calibration must not read test labels.
    missing = root/'test/labels/fall.txt'; old = missing.read_text(); missing.unlink()
    validation = run(config, weights, 'calibrate', root/'cal', model_factory=model)
    assert validation['fall_f1'] == 1
    assert all('/valid/' in c['source'].replace('\\', '/') for c in calls)
    assert all(c['conf'] == .001 and c['batch'] == 1 and c['rect'] is False for c in calls)
    frozen = root/'cal/calibration.json'; original = frozen.read_bytes()
    missing.write_text(old)
    # Test must not reread validation or retune using its labels.
    (root/'valid/labels/fall.txt').unlink()
    result = run(config, weights, 'test', root/'test-out', frozen, model)
    assert result['fall_f1'] == 1 and result['sample_count'] == 4
    assert frozen.read_bytes() == original
    rows = list(csv.DictReader((root/'test-out/predictions.csv').open()))
    assert next(r for r in rows if r['image'] == 'fall.png')['outcome'] == 'TP'
    assert (root/'test-out/error_review.md').is_file()
    assert 'threshold_search.csv' not in [p.name for p in (root/'test-out').iterdir()]


def test_changed_weights_and_duplicate_images_rejected(project):
    root, config, data, weights, model, calls = project
    run(config, weights, 'calibrate', root/'cal', model_factory=model)
    frozen = root/'cal/calibration.json'
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Weights differ'):
        run(config, weights, 'test', root/'out', frozen, model)
    weights.write_bytes(b'fake trained model for tests only')
    (root/'test/images/fall.png').write_bytes((root/'valid/images/fall.png').read_bytes())
    with pytest.raises(ValueError, match='Identical image'):
        run(config, weights, 'test', root/'out', frozen, model)
    assert not (root/'out').exists()


def test_test_requires_calibration_and_rejects_changed_image_size(project):
    root, config, data, weights, model, calls = project
    with pytest.raises(ValueError, match='requires'):
        run(config, weights, 'test', root/'out', model_factory=model)
    run(config, weights, 'calibrate', root/'cal', model_factory=model)
    config.write_text(yaml.safe_dump({'data_yaml': str(data), 'image_size': 320}))
    with pytest.raises(ValueError, match='Image size'):
        run(config, weights, 'test', root/'out', root/'cal/calibration.json', model)


def test_missing_labels_and_ambiguous_background_are_not_silently_relabelled(project):
    root, config, data, weights, model, calls = project
    _, rows = read_split(data, 'val')
    assert next(r for r in rows if r['image'] == 'none.png')['negative_group'] == 'no_person_label'
    assert next(r for r in rows if r['image'] == 'empty.png')['negative_group'] == 'empty_annotation'
    assert next(r for r in rows if r['image'] == 'fall.png')['true_fall'] is True
    (root/'valid/labels/empty.txt').unlink()
    with pytest.raises(FileNotFoundError, match='Missing annotation'):
        read_split(data, 'val')


def test_reject_wrong_class_mapping_and_existing_outputs(project):
    root, config, data, weights, model, calls = project
    class Wrong(model):
        def __init__(self, path):
            self.names = {0: 'person'}
    with pytest.raises(ValueError, match='class IDs'):
        run(config, weights, 'calibrate', root/'out', model_factory=Wrong)
    assert not (root/'out').exists()
    (root/'out').mkdir()
    with pytest.raises(FileExistsError):
        run(config, weights, 'calibrate', root/'out', model_factory=model)


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError, match='scores'):
        metrics([sample(True, score)], .25)
