import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, logits, labels):
        
        logits = logits - np.max(logits, axis=1, keepdims=True) 
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probs = probs
        self.labels = labels

        log_likelihood = -np.log(probs[np.arange(len(labels)), labels])
        loss = np.mean(log_likelihood)
        return loss

    def backward(self):

        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size
        return grad
