import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_history.csv")

plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.plot(df["epoch"], df["val_iou"], label="Val IoU")
plt.xlabel("Epoch")
plt.ylabel("Loss / IoU")
plt.legend()
plt.grid()
plt.show()