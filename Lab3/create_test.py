import os
import random
import shutil
from pathlib import Path

def split_to_test(image_dir, label_dir, test_image_dir, test_label_dir, test_percent=0.1):
    image_paths = list(Path(image_dir).glob('*.jpg'))
    num_to_move = int(len(image_paths) * test_percent)
    images_to_move = random.sample(image_paths, num_to_move)

    for img_path in images_to_move:
        label_path = Path(label_dir) / (img_path.stem + '.txt')

        shutil.move(str(img_path), str(test_image_dir / img_path.name))

        if label_path.exists():
            shutil.move(str(label_path), str(test_label_dir / label_path.name))

    print(f"Moved {len(images_to_move)} images from {image_dir} to {test_image_dir}")

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == '__main__':

    base_dir = Path(__file__).resolve().parent / 'dataset'
    test_percent = 0.1  

    paths = {
        'train_images': base_dir / 'images/train',
        'val_images': base_dir / 'images/val',
        'train_labels': base_dir / 'labels/train',
        'val_labels': base_dir / 'labels/val',
        'test_images': base_dir / 'images/test',
        'test_labels': base_dir / 'labels/test',
    }

    ensure_dirs(paths['test_images'], paths['test_labels'])

    split_to_test(paths['train_images'], paths['train_labels'], paths['test_images'], paths['test_labels'], test_percent)
    split_to_test(paths['val_images'], paths['val_labels'], paths['test_images'], paths['test_labels'], test_percent)
