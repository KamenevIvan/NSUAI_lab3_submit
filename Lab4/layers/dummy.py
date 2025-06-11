from .tensor import Tensor

class DummyMax:
    def __init__(self):
        self.inputs = []

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor, training=True):
        out = Tensor(x.data.copy(), requires_grad=x.requires_grad)
        out.set_creator(self)
        self.inputs = [x]
        self.output = out
        return out

    def backward(self, grad_output):
        return [grad_output]
