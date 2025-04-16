import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from dataset import RoadSegDataset
import segmentation_models_pytorch as smp
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(),  
    ToTensorV2()
])


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

    with torch.no_grad():
        for i in range(min(10, len(test_dataset))):  
            image, _ = test_dataset[i]
            image_input = image.unsqueeze(0).to(DEVICE)
            pred = model(image_input)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            visualize(image, pred[0], f"sample_{i}")

if __name__ == "__main__":
    main()
