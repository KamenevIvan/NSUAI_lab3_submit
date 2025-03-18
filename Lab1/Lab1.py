import os
import re

from tqdm import tqdm
from PIL import Image
import torch.optim as optim
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.models as models


def load_test_image(image_path, transform):
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)


def get_character_name(filename):
    match = re.match(r"([a-z_]+)_\d+", filename) 
    return match.group(1) if match else None


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss/len(train_loader), acc=100.0 * correct / total)

        validate(model, val_loader, criterion)

def validate(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100.0 * correct / total:.2f}%")


if __name__ == '__main__':
    train_dir = "dataset/simpsons_dataset"


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


    dataset = datasets.ImageFolder(root=train_dir, transform=transform)


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


    class_names = dataset.classes
    print(f"Классы ({len(class_names)}): {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA доступна! Используется устройство: {device}")
    test_dir = "dataset/kaggle_simpson_testset/kaggle_simpson_testset"


    test_images = []
    test_labels = []

    for filename in os.listdir(test_dir):
        char_name = get_character_name(filename)
        if char_name in class_names: 
            test_images.append(os.path.join(test_dir, filename))
            test_labels.append(class_names.index(char_name)) 

    print(f"Загружено тестовых изображений: {len(test_images)}")


    model = models.resnet34(pretrained=True)


    num_features = model.fc.in_features
    num_classes = len(class_names) 
    model.fc = nn.Linear(num_features, num_classes)


    model = model.to(device)


    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  


    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

    torch.save(model.state_dict(), "simpsons_resnet34.pth")


