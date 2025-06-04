import numpy as np
from .tensor import Tensor

class ReLU:
    def forward(self, x, training=True): 
        x_data = x.data
        self.mask = (x_data > 0)
        out_data = x_data * self.mask
        out = Tensor(out_data, requires_grad=True)
        out.set_creator(self)
        self.inputs = [x]
        return out

    def backward(self, dout):
        dx = dout * self.mask
        return [dx]
