import numpy as np

class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad.fill(0.0)

class SGD(Optimizer):
    def step(self):
        for param in self.params:
            param.value -= self.lr * param.grad
            #print(np.linalg.norm(param.grad), np.linalg.norm(param.value))