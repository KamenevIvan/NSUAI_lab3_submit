import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class RoadSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tiff", ".tif"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        if img_name.lower().endswith('.tiff'):
            mask_name = img_name[:-5] + '.tif'
        else:
            mask_name = img_name

        mask_path = os.path.join(self.mask_dir, mask_name)


        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)


        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']              
            mask = augmented['mask'].unsqueeze(0)   


        mask = mask.float() / 255.0

        return image, mask
