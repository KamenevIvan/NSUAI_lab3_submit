import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from model import MLP
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

    for c in range(num_classes):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))
        total = np.sum(y_true == c)

        precision[c] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[c] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        class_accuracy[c] = TP / total if total > 0 else 0.0

    return precision, recall, class_accuracy


def load_mnist_split():
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    X = X / 255.0  # нормализация к [0,1]

    # Разделим: сначала train_val и test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )
    # Теперь train и val из train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=10000, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# 2. Батч итератор
def batch_iter(X, y, batch_size=64, shuffle=True):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]

# 3. Обучение
def train():
    # Гиперпараметры
    input_dim = 784
    hidden_dims = [128, 64]
    output_dim = 10
    lr = 0.1
    epochs = 10
    batch_size = 64

    # Инициализация модели и лосса
    model = MLP(input_dim, hidden_dims, output_dim)
    loss_fn = SoftmaxCrossEntropyLoss()

    # Загрузка данных
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_split()

    for epoch in range(1, epochs + 1):
        model.train_mode = True
        train_loss = 0
        train_acc = 0
        n_train = 0

        # Тренировочный цикл
        for X_batch, y_batch in batch_iter(X_train, y_train, batch_size):
            logits = model.forward(X_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            dlogits = loss_fn.backward()

            model.backward(dlogits)
            model.step(lr)

            train_loss += loss * X_batch.shape[0]
            preds = np.argmax(logits, axis=1)
            train_acc += np.sum(preds == y_batch)
            n_train += X_batch.shape[0]

        train_loss /= n_train
        train_acc /= n_train

        # Валидация
        model.train_mode = False
        val_logits = model.forward(X_val, training=False)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == y_val)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    model.train_mode = False
    test_logits = model.forward(X_test, training=False)
    test_preds = np.argmax(test_logits, axis=1)
    test_acc = np.mean(test_preds == y_test)

    precision, recall, class_acc = compute_metrics(y_test, test_preds)

    print(f"\nTest Accuracy: {test_acc:.4f}")

    print("\nPer-class Precision:")
    print(np.round(precision, 3))

    print("\nPer-class Recall:")
    print(np.round(recall, 3))

    print("\nPer-class Accuracy:")
    print(np.round(class_acc, 3))

    show_predictions(X_test, y_test, test_preds, num_images=10)

if __name__ == "__main__":
    train()
