"""Consistent per-method command-line interface."""
import argparse
import importlib


def run(method, default_config):
    parser = argparse.ArgumentParser(description=f'{method} fall detection')
    parser.add_argument('action', choices=['train', 'evaluate', 'detect'])
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--source', default='0')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--no-show', action='store_true')
    if method != 'yolov8':
        parser.add_argument('--labels', help='Optional annotations for an image; otherwise use automatic person localisation')
    args = parser.parse_args()
    module = importlib.import_module(f'{method}.pipeline')
    if args.action == 'train':
        module.train(args.config)
    elif args.action == 'evaluate':
        module.evaluate(args.config, args.split)
    else:
        options = {'show': not args.no_show}
        if method != 'yolov8':
            options['labels'] = args.labels
        module.detect(args.config, args.source, **options)
