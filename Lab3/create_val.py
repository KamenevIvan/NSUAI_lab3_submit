import os
import shutil
import random
from glob import glob

images_dir = 'images/train'
labels_dir = 'labels/train'
output_images_train = 'images/train'
output_images_val = 'images/val'
output_labels_train = 'labels/train'
output_labels_val = 'labels/val'

os.makedirs(output_images_val, exist_ok=True)
os.makedirs(output_labels_val, exist_ok=True)

image_paths = glob(os.path.join(images_dir, '*.jpg'))
random.seed(42)
random.shuffle(image_paths)

split_index = int(len(image_paths) * 0.8)
train_images = image_paths[:split_index]
val_images = image_paths[split_index:]

def move_files(image_list, dest_img_dir, dest_lbl_dir):
    for image_path in image_list:
        filename = os.path.basename(image_path)
        label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

        shutil.move(image_path, os.path.join(dest_img_dir, filename))

        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(dest_lbl_dir, os.path.basename(label_path)))
        else:
            open(os.path.join(dest_lbl_dir, filename.replace('.jpg', '.txt')), 'w').close()

move_files(train_images, output_images_train, output_labels_train)
move_files(val_images, output_images_val, output_labels_val)

print(f"Done: {len(train_images)} train / {len(val_images)} val")
