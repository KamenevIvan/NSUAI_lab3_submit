import csv
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import MLP
from layers.optim import SGD
from activations.softmax_crossentropy import SoftmaxCrossEntropyLoss

import matplotlib.pyplot as plt


def show_predictions(X, y_true, y_pred, num_images=10):
    indices = np.random.choice(len(X), num_images, replace=False)
    X_samples = X[indices]
    y_true_samples = y_true[indices]
    y_pred_samples = y_pred[indices]

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_samples[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true_samples[i]}\nPred: {y_pred_samples[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_metrics(y_true, y_pred, num_classes=10):
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    class_accuracy = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))
        total = np.sum(y_true == c)

        precision[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        class_accuracy[c] = TP / total if total > 0 else 0.0

        if precision[c] + recall[c] > 0:
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        else:
            f1[c] = 0.0

    return precision, recall, class_accuracy, f1


def load_mnist_split():
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    X = X / 255.0 

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=10000, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def batch_iter(X, y, batch_size=64, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]


def train():
    
    input_dim = 784
    hidden_dims = [128, 64]
    output_dim = 10
    lr = 0.1
    epochs = 40
    batch_size = 64

    loadW = False
    weightsPath = "model_weights.pkl"

    model = MLP(input_dim, hidden_dims, output_dim)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)

    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_split()
    if not loadW:
        with open('metrics.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc'])
            for epoch in range(1, epochs + 1):
                model.train_mode = True
                train_loss = 0
                train_acc = 0
                n_train = 0

                for X_batch, y_batch in batch_iter(X_train, y_train, batch_size):
                    
                    #print(f"Форма X_batch: {X_batch.shape}, Тип: {type(X_batch)}")
                    optimizer.zero_grad()
                    logits = model.forward(X_batch, training=True)
                    #print(f"Форма logits.data: {logits.data.shape}, Тип: {type(logits)}")
                    loss = loss_fn.forward(logits.data, y_batch)
                    dlogits = loss_fn.backward()

                    model.backward(dlogits)
                    optimizer.step()    

                    train_loss += loss * X_batch.shape[0]
                    preds = np.argmax(logits.data, axis=1)
                    train_acc += np.sum(preds == y_batch)
                    n_train += X_batch.shape[0]

                train_loss /= n_train
                train_acc /= n_train

                model.train_mode = False
                val_logits = model.forward(X_val, training=False)
                val_preds = np.argmax(val_logits.data, axis=1)
                val_acc = np.mean(val_preds == y_val)

                writer.writerow([epoch, train_loss, train_acc, val_acc])
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
            
            model.save_weights(weightsPath)
            print("Model weights saved to model_weights.pkl")
    
    if loadW:   
        model.load_weights(weightsPath)
        print("Weights loaded.")

    model.train_mode = False
    test_logits = model.forward(X_test, training=False)
    test_preds = np.argmax(test_logits.data, axis=1)
    test_acc = np.mean(test_preds == y_test)

    precision, recall, class_acc, f1 = compute_metrics(y_test, test_preds)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    print("\nPer-class Precision:", np.round(precision, 3))
    print("Per-class Recall:", np.round(recall, 3))
    print("Per-class Accuracy:", np.round(class_acc, 3))
    print("Per-class F1:", np.round(f1, 3))

    show_predictions(X_test, y_test, test_preds, num_images=10)

if __name__ == "__main__":
    train()
