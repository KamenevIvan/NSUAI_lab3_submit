import numpy as np
from .tensor import Tensor  
from .params import Param

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = Param(np.ones((num_features,), dtype=np.float32))
        self.beta = Param(np.zeros((num_features,), dtype=np.float32))
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)

    def forward(self, x: Tensor, training=True):
        x_data = x.data.astype(np.float32)
        self.x = x  
        self.training = training

        if training:
            self.batch_mean = x_data.mean(axis=0)
            self.batch_var = x_data.var(axis=0)

            self.x_mu = x_data - self.batch_mean
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
            self.x_hat = self.x_mu * self.std_inv

            out_data = self.gamma.value * self.x_hat + self.beta.value

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            )
        else:
            x_hat = (x_data - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out_data = self.gamma.value * x_hat + self.beta.value

        out = Tensor(out_data, requires_grad=True)
        out.set_creator(self)
        self.inputs = [x]
        return out

    def backward(self, dout):
        x = self.x
        N, D = dout.shape

        self.beta.grad = np.sum(dout, axis=0)
        self.gamma.grad = np.sum(dout * self.x_hat, axis=0)

        dx_hat = dout * self.gamma.value
        dvar = np.sum(dx_hat * self.x_mu, axis=0) * -0.5 * self.std_inv**3
        dmu = np.sum(dx_hat * -self.std_inv, axis=0) + dvar * np.mean(-2.0 * self.x_mu, axis=0)

        dx = dx_hat * self.std_inv + dvar * 2.0 * self.x_mu / N + dmu / N
        return [dx]
    
    def parameters(self):
        return [self.gamma, self.beta]