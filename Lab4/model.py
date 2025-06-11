from layers.batchnorm import BatchNorm1d
from layers.linear import Linear
from layers.relu import ReLU
from layers.tensor import Tensor
from layers.dummy import DummyMax

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

        self.dummy = DummyMax()

        self.layers = [
            self.fc1, self.bn1, self.relu1,
            self.fc2, self.bn2, self.relu2,
            self.fc3, self.dummy
        ]

    def forward(self, x, training=True):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        #print(f"Вход модели: {x.data.shape}")
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, training=training)
            #print(f"После слоя {type(layer).__name__} ({i+1}): {x.data.shape}")
        return x

    # def backward(self, dout):
    #     for layer in reversed(self.layers):
    #         dout = layer.backward(dout)
    #     return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params
    

    def state_dict(self):
        weights = []
        for layer in self.layers:
            params = {}
            for attr in ['W', 'b', 'gamma', 'beta', 'running_mean', 'running_var']:
                if hasattr(layer, attr):
                    params[attr] = getattr(layer, attr)
            weights.append(params)
        return weights

    def load_state_dict(self, state_dict):
        for layer, params in zip(self.layers, state_dict):
            for key, value in params.items():
                setattr(layer, key, value)


def save_weights(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model.state_dict(), f)

def load_weights(model, filepath):
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
