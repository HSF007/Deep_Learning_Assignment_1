import numpy as np

class SGD:
    def __init__(self, eta=0.1):
        self.eta = eta
        
    def do_update(self, weights, biases, weight_grad, bias_grad):
        weights -= self.eta * weight_grad
        biases -= self.eta * bias_grad.reshape(biases.shape)
        return weights, biases

class Momentum:
    def __init__(self, eta=0.1, beta=0.5):
        self.eta = eta
        self.beta = beta
        self.prev_uw, self.prev_ub = None, None
    
    def do_update(self, weights, biases, weight_grad, bias_grad):
        if not self.prev_uw:
            self.prev_uw = np.zeros_like(weights)
        if not self.prev_ub:
            self.prev_ub = np.zeros_like(biases)
        
        self.prev_uw = self.beta * self.prev_uw + self.eta * weight_grad
        self.prev_ub = self.beta * self.prev_ub + self.eta * bias_grad.reshape(biases.shape)

        weights -= self.prev_uw
        biases -= self.prev_ub

        return weights, biases

class NAG:
    def __init__(self, eta=0.1, beta=0.5):
        self.eta = eta
        self.beta = beta
    
    def do_update(self, weights, biases, prev_vw, prev_vb, dw, db):
        
        prev_vw = self.beta*prev_vw + self.eta*dw
        prev_vb = self.beta*prev_vb + self.eta*db
        
        weights -= prev_vw 
        biases -= prev_vb
        return weights, biases, prev_vw, prev_vb

class RMSProp:
    def __init__(self):
        pass

class adam:
    def __init__(self):
        pass

class Nadam:
    def __init__(self):
        pass