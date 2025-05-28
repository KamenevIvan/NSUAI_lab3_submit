import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True, creator=None):
        if isinstance(data, memoryview):
            data = np.asarray(data, dtype=np.float32)
        if isinstance(data, np.ndarray) and data.dtype == np.float32:
            self.data = data
        else:
            self.data = np.array(data, dtype=np.float32)
            
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self.creator = creator 
        self._backward = lambda: None  

    def set_creator(self, creator):
        self.creator = creator

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        self.grad += grad

        self._backward()

        if self.creator is not None:
            for inp in self.creator.inputs:
                if inp.requires_grad:
                    inp.backward(inp.creator.output_grads[inp])

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"