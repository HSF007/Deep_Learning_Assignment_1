import numpy as np

class SGD:
    def __init__(self, eta=0.1, weight_decay=0):
        self.eta = eta
        self.weight_decay = weight_decay
        
    def do_update(self, weights, biases, weight_grad, bias_grad):
        for i in range(len(weights)):
            weights[i] -= self.eta * weight_grad[i]
            biases[i] -= self.eta * bias_grad[i]

            if self.weight_decay > 0:
                weights[i] -= self.eta * self.weight_decay * weights[i]
                biases[i] -= self.eta * self.weight_decay * biases[i]

        return weights, biases

class Momentum:
    def __init__(self, eta=0.1, beta=0.5, weight_decay=0):
        self.eta = eta
        self.beta = beta
        self.weight_decay = weight_decay
        self.prev_uw, self.prev_ub = None, None
    
    def do_update(self, weights, biases, weight_grad, bias_grad):
        if self.prev_uw is None:
            self.prev_uw = [np.zeros_like(w) for w in weights]
        if self.prev_ub is None:
            self.prev_ub = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.prev_uw[i] = self.beta * self.prev_uw[i] + self.eta * weight_grad[i]
            self.prev_ub[i] = self.beta * self.prev_ub[i] + self.eta * bias_grad[i]

            weights[i] -= self.prev_uw[i]
            biases[i] -= self.prev_ub[i]

            if self.weight_decay > 0:
                weights[i] -= self.eta * self.weight_decay * weights[i]
                biases[i] -= self.eta * self.weight_decay * biases[i]

        return weights, biases

class NAG:
    def __init__(self, eta=0.1, beta=0.5, weight_decay=0):
        self.eta = eta
        self.beta = beta
        self.weight_decay = weight_decay
    
    def do_update(self, weights, biases, prev_vw, prev_vb, dw, db):

        for i in range(len(weights)):
            prev_vw[i] = self.beta*prev_vw[i] + self.eta*dw[i]
            prev_vb[i] = self.beta*prev_vb[i] + self.eta*db[i]
        
            weights[i] -= prev_vw[i]
            biases[i] -= prev_vb[i]

            if self.weight_decay > 0:
                weights[i] -= self.eta * self.weight_decay * weights[i]
                biases[i] -= self.eta * self.weight_decay * biases[i]

        return weights, biases, prev_vw, prev_vb

class RMSProp:
    def __init__(self, eta=0.1, beta=0.5, eps=1e-6, weight_decay=0):
        self.vw = None
        self.vb = None
        self.eta = eta
        self.eps = eps
        self.beta = beta
        self.weight_decay = weight_decay
    
    def do_update(self, weights, biases, dw, db):
        if self.vw is None:
            self.vw = [np.zeros_like(w) for w in weights]
        if self.vb is None:
            self.vb = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.vw[i] = self.beta*self.vw[i] + (1 - self.beta)*(np.square(dw[i]))
            self.vb[i] = self.beta*self.vb[i] + (1 - self.beta)*(np.square(db[i]))

            weights[i] -= self.eta*dw[i]/(np.sqrt(self.vw[i]) + self.eps)
            biases[i] -= self.eta*db[i]/(np.sqrt(self.vb[i]) + self.eps)

            if self.weight_decay > 0:
                weights[i] -= self.eta * self.weight_decay * weights[i]/(np.sqrt(self.vw[i]) + self.eps)
                biases[i] -= self.eta * self.weight_decay * biases[i]/(np.sqrt(self.vb[i]) + self.eps)

        return weights, biases

class Adam:
    def __init__(self, eta, beta1=0.5, beta2=0.5, eps=1e-6, weight_decay = 0):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mw = None
        self.vw = None
        self.mb = None
        self.vb = None
        self.t = 0
        self.weight_decay = weight_decay
    
    def do_update(self, weights, biases, dw, db):
        if self.mw is None:
            self.mw = [np.zeros_like(w) for w in weights]
            self.vw = [np.zeros_like(w) for w in weights]
            self.mb = [np.zeros_like(b) for b in biases]
            self.vb = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        
        for i in range(len(weights)): 
            self.mw[i] = self.beta1*self.mw[i] + (1 - self.beta1)*dw[i]
            self.vw[i] = self.beta2*self.vw[i] + (1 - self.beta2)*(np.square(dw[i]))

            self.mb[i] = self.beta1*self.mb[i] + (1 - self.beta1)*db[i]
            self.vb[i] = self.beta2*self.vb[i] + (1- self.beta2)*(np.square(db[i]))

            biase_correction1 = 1 - np.power(self.beta1, self.t)
            biase_correction2 = 1 - np.power(self.beta2, self.t)

            weights[i] -= self.eta*(self.mw[i]/biase_correction1)/(np.sqrt(self.vw[i]/biase_correction2)+ self.eps)
            biases[i] -= self.eta*(self.mb[i]/biase_correction1)/(np.sqrt(self.vb[i]/biase_correction2) + self.eps)

            if self.weight_decay > 0:
                weights[i] -= self.eta * self.weight_decay * (weights[i])/(np.sqrt(self.vw[i]/biase_correction2)+ self.eps)
                biases[i] -= self.eta * self.weight_decay * (biases[i])/(np.sqrt(self.vb[i]/biase_correction2) + self.eps)

        return weights, biases


class Nadam:
    def __init__(self, eta=0.1, beta1=0.5, beta2=0.5, eps=1e-6, weight_decay=0):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mw, self.mb = None, None
        self.vw, self.vb = None, None
        self.t = 0
        self.weight_decay = weight_decay

    def do_update(self, weights, biases, dw, db):
        if self.mw is None:
            self.mw = [np.zeros_like(w) for w in weights]
            self.vw = [np.zeros_like(w) for w in weights]
            self.mb = [np.zeros_like(b) for b in biases]
            self.vb = [np.zeros_like(b) for b in biases]

        self.t += 1

        for i in range(len(weights)):
            self.mw[i] = self.beta1*self.mw[i] + (1 - self.beta1)*dw[i]
            self.mb[i] = self.beta1*self.mb[i] + (1 - self.beta1)*db[i]

            self.vw[i] = self.beta2*self.vw[i] + (1 - self.beta2)*(np.square(dw[i]))
            self.vb[i] = self.beta2*self.vb[i] + (1 - self.beta2)*(np.square(db[i]))


            biase_correction1 = 1 - np.power(self.beta1, self.t)
            biase_correction2 = 1 - np.power(self.beta2, self.t)

            weights[i] -= (self.eta/np.sqrt(self.vw[i]/biase_correction2 + self.eps))*((self.beta1*self.mw[i]/biase_correction1) + ((1-self.beta1)*dw[i]/biase_correction1))
            biases[i] -= (self.eta/np.sqrt(self.vb[i]/biase_correction2 + self.eps))*((self.beta1*self.mb[i]/biase_correction1) + ((1-self.beta1)*db[i]/biase_correction1))

            if self.weight_decay > 0:
                weights[i] -= (self.eta/np.sqrt(self.vw[i]/biase_correction2 + self.eps)) * self.weight_decay * weights[i]
                biases[i] -= (self.eta/np.sqrt(self.vb[i]/biase_correction2 + self.eps)) * self.weight_decay * biases[i]

        return weights, biases