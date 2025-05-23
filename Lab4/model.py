from layers.batchnorm import BatchNorm1d
from layers.linear import Linear
from layers.relu import ReLU

class MLP:
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        self.fc1 = Linear(input_dim, hidden_dims[0])
        self.bn1 = BatchNorm1d(hidden_dims[0])
        self.relu1 = ReLU()

        self.fc2 = Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = BatchNorm1d(hidden_dims[1])
        self.relu2 = ReLU()

        self.fc3 = Linear(hidden_dims[1], output_dim)
        self.layers = [self.fc1, self.bn1, self.relu1,
                       self.fc2, self.bn2, self.relu2,
                       self.fc3]  # fc3 – logits

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
        return dout  # градиент по входу (x)

    def step(self, lr):
        for layer in self.layers:
            if hasattr(layer, "step"):
                layer.step(lr)
