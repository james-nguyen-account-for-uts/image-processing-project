import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from hog_svm import pipeline


@pytest.fixture
def hog_dataset(tmp_path):
    for split in ('train', 'valid', 'test'):
        images = tmp_path/split/'images'
        labels = tmp_path/split/'labels'
        images.mkdir(parents=True)
        labels.mkdir()
        for name, class_id, vertical in [('fall', 1, False), ('up', 2, True)]:
            rows, cols = np.indices((128, 64))
            pixels = (((cols if vertical else rows)//8)%2*255).astype('uint8')
            Image.fromarray(pixels).save(images/f'{name}.png')
            (labels/f'{name}.txt').write_text(f'{class_id} 0.5 0.5 1 1\n')
    data = tmp_path/'data.yaml'
    data.write_text("train: train/images\nval: valid/images\ntest: test/images\nnames: ['0 person', 'fall', 'up']\n")
    config = tmp_path/'hog.yaml'
    config.write_text(f'data_yaml: {data.as_posix()}\noutput_dir: {(tmp_path/"output").as_posix()}\nc_values: [0.01, 0.1]\n')
    return config, tmp_path


def test_hog_fits_train_selects_val_and_evaluates_frozen_model(hog_dataset, monkeypatch):
    config, root = hog_dataset
    actual = pipeline.dataset
    calls = []
    def record(settings, split):
        calls.append(split)
        return actual(settings, split)
    monkeypatch.setattr(pipeline, 'dataset', record)
    pipeline.train(config)
    assert calls == ['train', 'val']
    model = root/'output/model.joblib'
    before = model.read_bytes()
    calls.clear()
    report = pipeline.evaluate(config)
    assert calls == ['test']
    assert report['fall_f1'] == 1.0
    assert model.read_bytes() == before
    classifier = pipeline.load_model(pipeline.settings(config))
    with Image.open(root/'test/images/fall.png') as image:
        prediction = pipeline.predict_boxes(image, [[0, 0, 64, 128]], classifier)
    assert prediction[0]['prediction'] == 'Fall'
    assert pipeline.predict_boxes(Image.new('RGB', (10, 10)), [], classifier) == []


def test_hog_headless_annotation_detection(hog_dataset):
    config, root = hog_dataset
    pipeline.train(config)
    result = pipeline.detect(config, str(root/'test/images/up.png'), str(root/'test/labels/up.txt'), show=False)
    output = Path(result['output_dir'])
    assert (output/'annotated.jpg').is_file()
    record = json.loads((output/'predictions.jsonl').read_text())
    assert record['status'] == 'No Fall'
    assert record['localisation'] == 'supplied_annotations'


def test_bad_crop_rejected():
    with pytest.raises(ValueError):
        pipeline.crop(Image.new('RGB', (20, 20)), [5, 5, 1, 1])
