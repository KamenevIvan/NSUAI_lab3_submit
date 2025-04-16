import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

import csv

import segmentation_models_pytorch as smp
from dataset import RoadSegDataset 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#  Конфигурация
CONFIG = {
    "model_name": "Unet",  # "Unet", "FPN", "DeepLabV3"
    "encoder_name": "resnet50",  # "efficientnet-b0", "mobilenet_v2"
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "classes": 1,
    "activation": None,
    "loss_name": "bce+dice",  # "bce", "dice", "bce+dice"
    "lr": 3e-4,
    "batch_size": 16,
    "num_epochs": 40,           #ПОПРОБОВАТЬ
    "img_dir": "tiff/train",
    "mask_dir": "tiff/train_labels",
    "val_img_dir": "tiff/val",
    "val_mask_dir": "tiff/val_labels",
    "checkpoint_path": "best_model.pth",
}

# Выбор модели
def get_model():
    model_class = getattr(smp, CONFIG["model_name"])
    return model_class(
        encoder_name=CONFIG["encoder_name"],
        encoder_weights=CONFIG["encoder_weights"],
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["classes"],
        activation=CONFIG["activation"],
    )


# def get_loss_fn():
#     if CONFIG["loss_name"] == "bce":
#         return nn.BCEWithLogitsLoss()
#     elif CONFIG["loss_name"] == "dice":
#         return smp.losses.DiceLoss(mode='binary')
#     elif CONFIG["loss_name"] == "bce+dice":
#         jaccard = smp.losses.JaccardLoss(mode='binary')
#         bce = smp.losses.SoftBCEWithLogitsLoss()
#         return smp.losses.JaccardLoss(mode='binary') + smp.losses.SoftBCEWithLogitsLoss()
#     else:
#         raise ValueError("Unknown loss")
    
def get_loss_fn():
    jaccard = smp.losses.JaccardLoss(mode='binary')
    bce = smp.losses.SoftBCEWithLogitsLoss()
    
    def _loss(y_pred, y_true):
        return jaccard(y_pred, y_true) + bce(y_pred, y_true)
    
    return _loss

# Аугментации и трансформы
train_transform = A.Compose([
    A.Resize(320, 320),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def test_dataloader():
    dataset = RoadSegDataset(CONFIG["img_dir"], CONFIG["mask_dir"], transform=train_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs, masks in loader:
        print("Image batch shape:", imgs.shape)
        print("Mask batch shape: ", masks.shape)
        break


def train():
    model = get_model().to(DEVICE)
    loss_fn =   get_loss_fn()    #smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7)  #smp.losses.JaccardLoss(mode="binary") #get_loss_fn()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    train_dataset = RoadSegDataset(CONFIG["img_dir"], CONFIG["mask_dir"], transform=train_transform)
    val_dataset = RoadSegDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    best_iou = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        total_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE).float()

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        iou_score = evaluate(model, val_loader)

        print(f"Loss: {avg_loss:.4f} | Val IoU: {iou_score:.4f}")

        # Save 
        if iou_score > best_iou:
            best_iou = iou_score
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            log_results(CONFIG, best_iou)

def evaluate(model, loader):
    model.eval()
    total_iou = 0.0
    plotted = False  
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            intersection = (preds * masks).sum((1, 2, 3))
            union = ((preds + masks) > 0).float().sum((1, 2, 3))
            iou = (intersection / (union + 1e-6)).mean().item()
            total_iou += iou

            # 
            # if not plotted:
            #     for i in range(min(2, images.size(0))):  
            #         plt.figure(figsize=(12,4))

            #         plt.subplot(1,3,1)
            #         plt.imshow(images[i].cpu().permute(1,2,0))
            #         plt.title("Image")

            #         plt.subplot(1,3,2)
            #         plt.imshow(masks[i].detach().cpu().numpy().squeeze(), cmap='gray')

            #         plt.title("Mask")

            #         plt.subplot(1,3,3)
            #         plt.imshow(preds[i].detach().cpu().numpy().squeeze(), cmap='gray')

            #         plt.title("Prediction")

            #         plt.show()
            #     plotted = True
            
    return total_iou / len(loader)

def log_results(config, best_iou):
    log_path = "training_log.csv"
    config_to_log = {k: v for k, v in config.items() if isinstance(v, (str, int, float))}
    config_to_log["best_iou"] = round(best_iou, 4)

    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config_to_log.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(config_to_log)


if __name__ == "__main__":
    train()
