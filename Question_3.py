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
        prev_vb = self.beta*prev_vb + self.eta*db.reshape(biases.shape)
        
        weights -= prev_vw 
        biases -= prev_vb
        return weights, biases, prev_vw, prev_vb

class RMSProp:
    def __init__(self, eta=0.1, beta=0.5, eps=1e-6):
        self.vw = None
        self.vb = None
        self.eta = eta
        self.eps = eps
        self.beta = beta
    
    def do_update(self, weights, biases, dw, db):
        if not self.vw:
            self.vw = np.zeros_like(weights)
        if not self.vb:
            self.vb = np.zeros_like(biases)
        
        vw = self.beta*self.vw + (1 - self.beta)*(np.square(dw))
        vb = self.beta*self.vb + (1 - self.beta)*(np.square(db.reshape(biases.shape)))

        weights -= self.eta*dw/(np.sqrt(vw) + self.eps)
        biases -= self.eta*db/(np.sqrt(vb) + self.eps)

        return weights, biases

class adam:
    def __init__(self, eta, beta1, beta2, eps):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mw = None
        self.vw = None
        self.mb = None
        self.vb = None
        self.t = 0
    
    def do_update(self, weights, biases, dw, db):
        if not self.mw:
            self.mw = np.zeros_like(weights)
            self.vw = np.zeros_like(weights)
            self.mb = np.zeros_like(biases)
            self.vb = np.zeros_like(biases)
        
        self.mw = self.beta1*self.mw + (1 - self.beta1)*dw
        self.vw = self.beta2*self.vw + (1 - self.beta2)*(np.square(dw))

        self.mb = self.beta1*self.mb + (1 - self.beta1)*db.reshape(biases.shape)
        self.vb = self.beta2*self.vb + (1- self.beta2)*(np.square(db.reshape(biases.shape)))

        self.t += 1

        weights -= self.eta*self.mw/((np.sqrt(self.vw/(1-np.power(self.beta2, self.t))) + self.eps)*(1- np.power(self.beta1, self.t)))
        biases -= self.eta*self.mb/((np.sqrt(self.vb/(1-np.power(self.beta2, self.t))) + self.eps)*(1- np.power(self.beta1, self.t)))

        return weights, biases


class Nadam:
    def __init__(self, eta=0.1, beta1=0.5, beta2=0.5, eps=1e-6):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mw, self.mb = None, None
        self.vw, self.vb = None, None
        self.t = 0

    def do_update(self, weights, biases, dw, db):
        if not self.mw:
            self.mw, self.vw = np.zeros_like(weights), np.zeros_like(weights)
            self.mb, self.vb = np.zeros_like(biases), np.zeros_like(biases)

        self.mw = self.beta1*self.mw + (1 - self.beta1)*dw
        self.mb = self.beta1*self.mb + (1 - self.beta1)*db.reshape(biases.shape)

        self.vw = self.beta2*self.vw + (1 - self.beta2)*(np.square(dw))
        self.vb = self.beta2*self.vb + (1 - self.beta2)*(np.square(db.reshape(biases.shape)))

        self.t += 1

        biase_correction1 = 1 - np.power(self.beta1, self.t)
        biase_correction2 = 1 - np.power(self.beta2, self.t)

        weights -= (self.eta/np.sqrt(self.vw/biase_correction2 + self.eps))*(self.beta1*self.mw/biase_correction1 + (1-self.beta1)*dw/biase_correction1)
        biases -= (self.eta/np.sqrt(self.vb/biase_correction2 + self.eps))*(self.beta1*self.mb/biase_correction1 + (1-self.beta1)*db/biase_correction1)

        return weights, biases