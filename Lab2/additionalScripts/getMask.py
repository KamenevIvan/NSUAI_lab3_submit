import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2


MODEL_PATH = "best_model.pth"  
IMAGE_PATH = "tiff/val/10228690_15.tiff"  
MASK_PATH = "tiff/val_labels/10228690_15.tif"   

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'


transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])


model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def load_image(path):
    image = np.array(Image.open(path).convert("RGB"))
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0)  

def load_mask(path):
    mask = np.array(Image.open(path).convert("L"))
    return mask / 255.0  # normalize to [0, 1]


image_tensor = load_image(IMAGE_PATH)
with torch.no_grad():
    pred = model(image_tensor).sigmoid().squeeze().numpy()


gt_mask = load_mask(MASK_PATH)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(np.array(Image.open(IMAGE_PATH)))
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(pred > 0.5, cmap='gray')
plt.title("Prediction > 0.5")

plt.tight_layout()
plt.show()
