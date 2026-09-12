# Dataset directory

Download Version 1 of the [Roboflow Fall dataset](https://universe.roboflow.com/search-kinvz/fall-ljaaq/dataset/1), export it in YOLO format, and extract it as:

```text
data/fall/
├── data.yaml
├── train/
├── valid/
└── test/        # May be absent if the export has no test split
```

Do not commit the downloaded images or labels.
