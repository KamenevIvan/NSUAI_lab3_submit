import numpy as np
from .tensor import Tensor

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x, training=True): 
        x_data = x.data if isinstance(x, Tensor) else x
        self.mask = (x_data > 0)
        out_data = x_data * self.mask
        out = Tensor(out_data, requires_grad=True)
        out.set_creator(self)
        
        def _backward():
            dx = out.grad * self.mask
            if isinstance(x, Tensor) and x.requires_grad:
                x.backward(dx)
        
        out._backward = _backward
        return out
    
    def backward(self, dout):
            dx = dout * self.mask  
            return dx