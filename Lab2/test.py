import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import RoadSegDataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

CONFIG = {
    "model_name": "Unet",
    "encoder_name": "resnet50",
    "encoder_weights": "imagenet",
    "in_channels": 3,
    "classes": 1,
    "activation": None,
    "checkpoint_path": "best_model.pth",
    "test_img_dir": "tiff/test",
    "mask_dir": "tiff/test_labels",
    "output_dir": "predictions",
    "batch_size": 8,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2()
])

def calculate_metrics(preds, targets):
    preds = preds.float()
    targets = targets.float()
    
    # Micro 
    inter = (preds * targets).sum()
    union = (preds + targets).sum()
    micro_iou = inter / (union - inter + 1e-6)
    
    # Macro 
    per_image_iou = []
    for p, t in zip(preds, targets):
        i = (p * t).sum()
        u = (p + t).sum()
        per_image_iou.append(i / (u - i + 1e-6))
    macro_iou = torch.mean(torch.stack(per_image_iou))
    
    # Dice 
    dice = (2 * inter) / (preds.sum() + targets.sum() + 1e-6)
    
    # Pixel accuracy
    correct = (preds == targets).sum()
    total = torch.numel(preds)
    pixel_acc = correct / total
    
    # Precision and Recall
    true_pos = inter
    false_pos = preds.sum() - inter
    false_neg = targets.sum() - inter
    
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    
    return {
        'micro_iou': micro_iou.item(),
        'macro_iou': macro_iou.item(),
        'dice': dice.item(),
        'pixel_acc': pixel_acc.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def get_model():
    model_class = getattr(smp, CONFIG["model_name"])
    return model_class(
        encoder_name=CONFIG["encoder_name"],
        encoder_weights=CONFIG["encoder_weights"],
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["classes"],
        activation=CONFIG["activation"],
    )

def visualize(image, mask_pred, fname):
    image_cpu = image.cpu()
    mask_pred_cpu = mask_pred.cpu()

    grid = make_grid([image_cpu, mask_pred_cpu.expand_as(image_cpu)], nrow=2)
    plt.figure(figsize=(6, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(fname)
    plt.savefig(os.path.join(CONFIG["output_dir"], fname + ".png"))
    plt.close()

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    model = get_model().to(DEVICE)
    model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=DEVICE))
    model.eval()

    test_dataset = RoadSegDataset(CONFIG["test_img_dir"], CONFIG["mask_dir"], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    metrics = {
        'micro_iou': 0.0,
        'macro_iou': 0.0,
        'dice': 0.0,
        'pixel_acc': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }
    total_samples = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            
            batch_metrics = calculate_metrics(preds, masks)
            
            
            batch_size = images.size(0)
            total_samples += batch_size
            for key in metrics:
                metrics[key] += batch_metrics[key] * batch_size
            
            
            if i == 0:
                for j in range(min(3, batch_size)):
                    visualize(images[j], preds[j], f"sample_{i}_{j}")


    for key in metrics:
        metrics[key] /= total_samples

    print("\nTest Metrics:")
    print(f"Micro IoU: {metrics['micro_iou']:.4f}")
    print(f"Macro IoU: {metrics['macro_iou']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"Pixel Accuracy: {metrics['pixel_acc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()