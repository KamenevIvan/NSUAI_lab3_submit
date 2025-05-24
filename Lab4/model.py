from layers.batchnorm import BatchNorm1d
from layers.linear import Linear
from layers.relu import ReLU

import pickle

class MLP:
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        self.fc1 = Linear(input_dim, hidden_dims[0])
        self.bn1 = BatchNorm1d(hidden_dims[0])
        self.relu1 = ReLU()

        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = BatchNorm1d(hidden_dims[1])
        self.relu2 = ReLU()

        self.fc3 = Linear(hidden_dims[1], output_dim)

        self.layers = [
            self.fc1, self.bn1, self.relu1,
            self.fc2, self.bn2, self.relu2,
            self.fc3
        ]

    def forward(self, x, training=True):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(lr)

    def save_weights(self, filepath):
        weights = []
        for layer in self.layers:
            params = {}
            for attr in ['W', 'b', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(layer, attr):
                    params[attr] = getattr(layer, attr)
            weights.append(params)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        for layer, params in zip(self.layers, weights):
            for key, value in params.items():
                setattr(layer, key, value)