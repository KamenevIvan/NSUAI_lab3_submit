import numpy as np
from layers.tensor import Tensor

class SoftmaxCrossEntropyOp:
    def __init__(self, logits, labels):
        self.inputs = [logits]
        self.labels = labels
        self.output_grads = [None]  

    def forward(self):
        logits = self.inputs[0].data

        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probs = probs
        self.batch_size = logits.shape[0]

        self.output_grads[0] = None  

        log_likelihood = -np.log(probs[np.arange(self.batch_size), self.labels])
        loss = np.mean(log_likelihood)

        out = Tensor(loss, requires_grad=True)
        out.set_creator(self)
        return out

    def backward(self, grad_output):
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.labels] -= 1
        grad /= self.batch_size
        return [grad * grad_output]  
    
def softmax_crossentropy(logits, labels):
    op = SoftmaxCrossEntropyOp(logits, labels)
    return op.forward()