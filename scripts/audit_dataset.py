"""Read-only dataset inventory and exact cross-split duplicate audit."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fall_detection.image_evaluation import read_split


def audit(data_yaml):
    splits, pixels, file_hashes = {}, defaultdict(list), defaultdict(list)
    for split in ('train', 'val', 'test'):
        names, rows = read_split(data_yaml, split)
        splits[split] = {'images': len(rows), 'positive_images': sum(r['true_fall'] for r in rows),
                         'groups': dict(Counter(r['negative_group'] for r in rows))}
        for row in rows:
            identity = {'split': split, 'image': row['image']}
            file_hashes[row['image_sha256']].append(identity)
            with Image.open(row['source']) as source:
                rgb = source.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            pixels[pixel_hash].append(identity)
    cross = lambda groups: [v for v in groups.values() if len({r['split'] for r in v}) > 1]
    return {'class_names': names, 'splits': splits, 'cross_split_file_duplicates': cross(file_hashes),
            'cross_split_identical_pixels': cross(pixels),
            'limitations': ['Does not establish the meaning of 0 person or completeness of labels.',
                            'Near-duplicates and same-recording frames require additional review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/fall/data.yaml')
    parser.add_argument('--output', default='runs/data_audit.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new audit output path')
    result = audit(args.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result['splits'], indent=2))
    print('Cross-split exact-pixel duplicate groups:', len(result['cross_split_identical_pixels']))


if __name__ == '__main__':
    main()
