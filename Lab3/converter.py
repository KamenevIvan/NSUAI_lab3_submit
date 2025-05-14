import pandas as pd
import os

csv_path = '_annotations.csv'
images_dir = 'images/train'
labels_dir = 'labels/train'
os.makedirs(labels_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for filename, group in df.groupby('filename'):
    image_path = os.path.join(images_dir, filename)
    if not os.path.exists(image_path):
        print(f"[!] Warning: image {filename} not found, skipping.")
        continue

    width = group.iloc[0]['width']
    height = group.iloc[0]['height']
    txt_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

    with open(txt_path, 'w') as f:
        for _, row in group.iterrows():
            x_center = ((row['xmin'] + row['xmax']) / 2) / width
            y_center = ((row['ymin'] + row['ymax']) / 2) / height
            box_width = (row['xmax'] - row['xmin']) / width
            box_height = (row['ymax'] - row['ymin']) / height
            f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")