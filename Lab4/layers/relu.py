import numpy as np

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
    

#Test run

x = np.array([
    [1.0, -0.5, 0.0],
    [-2.0, 3.0, 4.0],
])

relu = ReLU()
out = relu.forward(x)

print("Input:\n", x)
print("ReLU Output:\n", out)

dout = np.ones_like(x)
dx = relu.backward(dout)

print("Grad from next layer:\n", dout)
print("Backward grad through ReLU:\n", dx)