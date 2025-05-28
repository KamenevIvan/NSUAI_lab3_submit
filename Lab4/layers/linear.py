import numpy as np
from .params import Param
from .tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        W = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        b = np.zeros((out_features,))
        self.W = Param(W)
        self.b = Param(b)
        self.x = None

    def forward(self, x, training=True): 
        self.x = x.data if isinstance(x, Tensor) else x
        out_data = self.x @ self.W.value.T + self.b.value
        out = Tensor(out_data, requires_grad=True)
        out.set_creator(self)
        
        def _backward():
            batch_size = out.grad.shape[0]
            self.W.grad = (out.grad.T @ self.x) / batch_size
            self.b.grad = np.mean(out.grad, axis=0)
            dx = out.grad @ self.W.value
            if isinstance(x, Tensor) and x.requires_grad:
                x.backward(dx)
        
        out._backward = _backward
        return out
    
    def backward(self, dout):
        batch_size = dout.shape[0]
        self.W.grad = (dout.T @ self.x) / batch_size
        self.b.grad = np.mean(dout, axis=0)
        dx = dout @ self.W.value
        return dx

    def parameters(self):
        return [self.W, self.b]