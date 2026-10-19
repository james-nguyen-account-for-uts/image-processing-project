import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import numpy as np

from common.comparison import load_reports, display
from yolov8.report import save_report


def configs(root):
    (root/'configs').mkdir()
    for file, text in [('geometry.yaml','output_dir: runs/geometry'),
                       ('hog_svm.yaml','output_dir: runs/hog_svm'),
                       ('default.yaml','project_dir: runs/yolov8')]:
        (root/'configs'/file).write_text(text)


def test_yolo_report_preserves_class_mapping_and_transposes_matrix(tmp_path):
    metrics = SimpleNamespace(names={0:'0 person',1:'fall',2:'up'},
        box=SimpleNamespace(ap_class_index=np.array([1,2]),p=[.8,.9],r=[.6,.7],f1=[.6857,.7875],map50=.75,map=.5),
        confusion_matrix=SimpleNamespace(matrix=np.arange(16).reshape(4,4)))
    path=save_report(metrics,tmp_path,'test','weights.pt','data.yaml')
    report=json.loads(path.read_text())
    assert report['per_class']['fall']['precision']==.8
    assert report['confusion_matrix']['counts'][1][0]==1
    assert report['confusion_matrix']['labels']==['0 person','fall','up','background']
    assert report['split']=='test'


def test_missing_results_render_without_training(tmp_path):
    configs(tmp_path)
    assert all('error' in r for r in load_reports(tmp_path))
    assert display(tmp_path,show=False).is_file()


def test_selected_geometry_rule_and_corrupt_report(tmp_path):
    configs(tmp_path)
    scores={'fall_precision':.8,'fall_recall':.9,'fall_f1':.85,
            'confusion_matrix':{'labels':['No Fall','Fall'],'counts':[[10,2],[3,20]]}}
    path=tmp_path/'runs/geometry/evaluation/test/metrics.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'model':{'selected_rule_by_validation':'aspect_ratio_position'},
        'splits':{'test':{'rules':{'aspect_ratio':{},'aspect_ratio_position':scores}}}}))
    hog=tmp_path/'runs/hog_svm/evaluation/test/metrics.json'
    hog.parent.mkdir(parents=True)
    hog.write_text('broken')
    reports=load_reports(tmp_path)
    assert reports[0]['values']==[.8,.9,.85]
    assert 'Cannot read' in reports[1]['error']
    assert display(tmp_path,show=False).is_file()


def test_yolo_test_report_loads_and_renders_all_metrics(tmp_path):
    configs(tmp_path)
    metrics = SimpleNamespace(names={0:'0 person',1:'fall',2:'up'},
        box=SimpleNamespace(ap_class_index=[1,2],p=[.8,.9],r=[.6,.7],f1=[.6857,.7875],map50=.75,map=.5),
        confusion_matrix=SimpleNamespace(matrix=np.array([[0,0,0,0],[0,30,2,4],[0,3,20,2],[0,5,1,0]])))
    save_report(metrics,tmp_path/'runs/yolov8','test','weights.pt','data.yaml')
    reports=load_reports(tmp_path)
    assert reports[2]['values']==[.8,.6,.6857,.75,.5]
    assert 'error' not in reports[2]
    assert display(tmp_path,show=False).is_file()
