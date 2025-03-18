import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(num_classes=42)  
model.load_state_dict(torch.load("simpsons_resnet34.pth"))  
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class_names = sorted(os.listdir("dataset/simpsons_dataset")) 
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}

def predict_character(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return class_names[predicted_class] 

test_dir = "dataset/test"

test_images = []
test_labels = []

for character_name in os.listdir(test_dir):
    character_path = os.path.join(test_dir, character_name)

    if os.path.isdir(character_path) and character_name in class_names:
        for filename in os.listdir(character_path):
            image_path = os.path.join(character_path, filename)
            test_images.append(image_path)
            test_labels.append(character_name) 

y_true = []
y_pred = []

for image, label in zip(test_images, test_labels):
    predicted_character = predict_character(image)
    y_true.append(label)
    y_pred.append(predicted_character)


correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
total = len(y_true)
accuracy = 100 * correct / total
print(f"Overall Accuracy: {accuracy:.2f}%")


report = classification_report(y_true, y_pred, target_names=class_names)
print(report)


y_true_np = np.array([class_to_idx[label] for label in y_true])
y_pred_np = np.array([class_to_idx[label] for label in y_pred])

class_accuracies = {}
for class_name, class_idx in class_to_idx.items():
    mask = y_true_np == class_idx
    if mask.sum() > 0:
        class_accuracy = (y_pred_np[mask] == class_idx).sum() / mask.sum()
        class_accuracies[class_name] = class_accuracy * 100 

print("\nAccuracy per class:")
for class_name, acc in class_accuracies.items():
    print(f"{class_name}: {acc:.2f}%")


with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Overall Accuracy: {accuracy:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Accuracy per class:\n")
    for class_name, acc in class_accuracies.items():
        f.write(f"{class_name}: {acc:.2f}%\n")

print("\n Результаты сохранены в test_results.txt")
