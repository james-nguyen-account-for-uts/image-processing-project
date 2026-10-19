# Dataset placement

Extract the Roboflow Fall v1 YOLO export directly into this directory:

```text
data.yaml
train/images/  train/labels/
valid/images/  valid/labels/
test/images/   test/labels/
```

`data.yaml` must use `train: train/images`, `val: valid/images`, and `test: test/images` (relative to this directory), with names `['0 person', 'fall', 'up']`. Every image needs a matching YOLO label file. Dataset content is intentionally ignored by Git. All three experiment configs use `data/data.yaml`.
