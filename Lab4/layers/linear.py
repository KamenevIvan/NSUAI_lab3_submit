import numpy as np

class Linear:
    def __init__(self, in_features, out_features):

        self.W = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.b = np.zeros((out_features,))

        self.x = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        """
        dout: градиенты от следующего слоя, shape (batch_size, out_features)
        Возвращает градиенты по входу: (batch_size, in_features)
        """
        batch_size = dout.shape[0]

        self.dW = (dout.T @ self.x) / batch_size
        self.db = np.mean(dout, axis=0)

        dx = dout @ self.W  
        return dx

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db