"""Run with: python main.py (from any working directory)."""
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
METHODS = {'1': ('Bounding box', 'bounding_box.pipeline'),
           '2': ('HOG–SVM', 'hog_svm.pipeline'),
           '3': ('YOLOv8', 'yolov8.pipeline')}


def main():
    os.chdir(ROOT)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"runs/.ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"runs/.matplotlib"))
    while True:
        print('\nFall detection\n1. Bounding box\n2. HOG–SVM\n3. YOLOv8\n4. Display evaluation charts\n0. Exit')
        try:
            choice = input('Choose 1, 2, 3, 4, or exit: ').strip().lower()
            if choice in ('0', 'exit', 'q'):
                return
            if choice == '4':
                importlib.import_module('common.comparison').display(ROOT)
                continue
            if choice not in METHODS:
                print('Please enter 1, 2, 3, 4, or exit.')
                continue
            name, module_name = METHODS[choice]
            print(f'\n{name}\n1. Train / calibrate\n2. Evaluate on test set\n3. Detect image, video, or webcam\n0. Back')
            action = input('Choose an action: ').strip().lower()
            if action in ('0', 'back'):
                continue
            if action in ('exit', 'q'):
                return
            if action not in ('1', '2', '3'):
                print('Invalid action.')
                continue
            module = importlib.import_module(module_name)
            if action == '1':
                module.train()
            elif action == '2':
                module.evaluate()
            else:
                source = input('Image/video path or camera index [0]: ').strip().strip('"') or '0'
                if choice != '3':
                    print('Automatic person boxes use pretrained YOLOv8; the selected method classifies posture.')
                    labels = input('Optional annotation file for an image [Enter for automatic boxes]: ').strip().strip('"') or None
                    module.detect(source=source, labels=labels)
                else:
                    module.detect(source=source)
        except EOFError:
            return
        except KeyboardInterrupt:
            print('\nOperation cancelled. Returning to menu.')
        except Exception as error:
            print(f'\nCould not complete operation: {error}')
            if isinstance(error, ImportError):
                print('Install dependencies using: python -m pip install -r requirements.txt')


if __name__ == '__main__':
    main()
