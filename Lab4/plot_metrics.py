import matplotlib.pyplot as plt
import csv

epochs, train_loss, train_acc, val_acc = [], [], [], []

with open('metrics.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        epochs.append(int(row['epoch']))
        train_loss.append(float(row['train_loss']))
        train_acc.append(float(row['train_acc']))
        val_acc.append(float(row['val_acc']))

#Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

#Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc', color='blue')
plt.plot(epochs, val_acc, label='Val Acc', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()