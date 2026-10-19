"""Image/video/camera localisation for region classifiers.

Automatic localisation uses pretrained YOLOv8 person boxes. Fall decisions are
made exclusively by the selected geometry or HOG–SVM classifier.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from common.dataset import parse_yolo_box


def annotation_boxes(path, data_yaml, width, height):
    names = yaml.safe_load(Path(data_yaml).read_text())['names']
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    boxes = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        class_id, cx, cy, w, h = parse_yolo_box(line, Path(path), number, names)
        if names[class_id] not in ('fall', 'up'):
            continue
        boxes.append([max(0, (cx-w/2)*width), max(0, (cy-h/2)*height),
                      min(width, (cx+w/2)*width), min(height, (cy+h/2)*height)])
    return boxes


def run(source, classify, output_dir, name, *, labels=None, data_yaml=None, show=True):
    import cv2
    source = str(source)
    still = Path(source).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    if labels is not None and not still:
        raise ValueError('Annotations can only be supplied for a still image')
    detector = None
    if labels is None:
        from ultralytics import YOLO
        detector = YOLO('yolov8n.pt')
    out = Path(output_dir)/'predictions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=True)
    capture = None
    count = 0
    try:
        if not still:
            capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open camera/video: {source}')
        with (out/'predictions.jsonl').open('w') as log:
            while True:
                frame = cv2.imread(source) if still else capture.read()[1]
                if frame is None:
                    if count == 0:
                        raise RuntimeError(f'Cannot read image/frame: {source}')
                    break
                height, width = frame.shape[:2]
                if labels is not None:
                    boxes = annotation_boxes(labels, data_yaml, width, height)
                else:
                    results = detector.predict(frame, classes=[0], conf=0.25, verbose=False)
                    boxes = [b for result in results if result.boxes is not None
                             for b in result.boxes.xyxy.cpu().tolist()]
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                regions = classify(image, boxes)
                status = ('Fall' if any(r['prediction']=='Fall' for r in regions) else 'No Fall') if regions else 'No person detected'
                record = {'frame': count, 'method': name, 'status': status, 'regions': regions,
                          'localisation': 'supplied_annotations' if labels else 'pretrained_yolov8n'}
                log.write(json.dumps(record)+'\n')
                for region in regions:
                    x1,y1,x2,y2 = map(round, region['bbox_xyxy'])
                    color = (0,0,255) if region['prediction']=='Fall' else (0,200,0)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, region['prediction'], (x1,max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                cv2.putText(frame, f'{name}: {status}', (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                count += 1
                if still:
                    if not cv2.imwrite(str(out/'annotated.jpg'),frame):
                        raise RuntimeError('Could not save annotated image')
                    print(status)
                if show:
                    cv2.imshow(name,frame)
                    key=cv2.waitKey(0 if still else 1)&0xff
                    if key in (ord('q'),27) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:
                        break
                if still:
                    break
    finally:
        if capture is not None:
            capture.release()
        if show:
            cv2.destroyAllWindows()
    print(f'Predictions saved to {out}')
    return {'frame_count': count, 'output_dir': str(out)}
