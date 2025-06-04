import numpy as np
from .params import Param
from .tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        W = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        b = np.zeros((out_features,))
        self.W = Param(W)
        self.b = Param(b)

    def forward(self, x, training=True): 
        out_data = x.data @ self.W.value.T + self.b.value
        out = Tensor(out_data, requires_grad=True)
        out.set_creator(self)
        self.inputs = [x]
        self.x = x 
        return out

    def backward(self, dout):
        x = self.x
        batch_size = dout.shape[0]

        self.W.grad = (dout.T @ x.data) / batch_size
        self.b.grad = np.mean(dout, axis=0)

        dx = dout @ self.W.value
        return [dx]  

    def parameters(self):
        return [self.W, self.b]
