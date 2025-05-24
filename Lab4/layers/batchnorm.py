import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((num_features,))
        self.beta = np.zeros((num_features,))

        self.running_mean = np.zeros((num_features,))
        self.running_var = np.ones((num_features,))

        self.x_hat = None
        self.std_inv = None
        self.x_mu = None

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            self.x_mu = x - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.eps)
            self.x_hat = self.x_mu * self.std_inv

            out = self.gamma * self.x_hat + self.beta


            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )
        else:
            
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, dout):
        N, D = dout.shape

        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * self.x_hat, axis=0)

        dx_hat = dout * self.gamma

        dvar = np.sum(dx_hat * self.x_mu, axis=0) * -0.5 * self.std_inv**3
        dmu = np.sum(dx_hat * -self.std_inv, axis=0) + dvar * np.mean(-2.0 * self.x_mu, axis=0)

        dx = dx_hat * self.std_inv + dvar * 2.0 * self.x_mu / N + dmu / N

        return dx

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta


#Test
'''
np.random.seed(0)
x = np.random.randn(4, 3) * 5 + 10
bn = BatchNorm1d(3)
out = bn.forward(x, training=True)
print("Normalized output:\n", out)

# Градиент
dout = np.random.randn(4, 3)
dx = bn.backward(dout)
print("dx:\n", dx)
'''