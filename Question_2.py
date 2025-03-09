import numpy as np
from Question_3 import *
import wandb

class NeuralNetwork:
    def __init__(self, input_features:int, hidden_layers:list[int], activation:str, output_features:int=10, weight_init:str="random"):
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_layers = hidden_layers
        self.weight_init = weight_init


        self.activation = self.get_activation(activation)
        self.activation_derivative = self.get_activation_derivative(activation)


        self.weights = []
        self.biases = []
        self.history_grad = []

        layer_sizes = [input_features] + hidden_layers + [output_features]

        for i in range(len(layer_sizes) - 1):
            if self.weight_init.lower() == "xavier":
                interval = np.sqrt(6/((layer_sizes[i] + layer_sizes[i+1])))
                w = np.random.uniform(-interval, interval, size=(layer_sizes[i], layer_sizes[i+1]))
                b = np.random.uniform(-interval, interval, (1, layer_sizes[i+1]))
            else:
                w = np.random.Generator.standard_normal(size=(layer_sizes[i], layer_sizes[i+1]))
                b = np.random.Generator.standard_normal(size=(1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.weights = self.weights
        self.biases = self.biases
        
    def get_activation(self, activation):
        if activation.lower() == "identity":
            return lambda x: x
        elif activation.lower() == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation.lower() == "tanh":
            return lambda x: np.tanh(x)
        elif activation.lower() == "relu":
            return lambda x: np.maximum(0, x)
        else:
            raise NameError(f"Error: No such activation as {activation}.\nYou can choose activation fucntions from [identity, sigmoid, tanh, ReLU].")
    
    def get_activation_derivative(self, activation):
        if activation.lower() == "identity":
            return lambda x: np.ones_like(x)
        elif activation.lower() == "sigmoid":
            return lambda x: x * (1 - x)
        elif activation.lower() == "tanh":
            return lambda x: 1 - x**2
        elif activation.lower() == "relu":
            return lambda x: (x > 0).astype(float)
        else:
            raise NameError(f"Error: No such activation as {activation}.\nYou can choose activation fucntions from [identity, sigmoid, tanh, ReLU].")
        
    def feedforward(self, X):
        self.active_values = [X]
        self.hidden_values = []

        for i in range(len(self.weights) - 1):
            z = np.dot(self.active_values[-1], self.weights[i]) + self.biases[i]
            self.hidden_values.append(z)
            data = self.activation(z)
            self.active_values.append(data)

        z = np.dot(self.active_values[-1], self.weights[-1]) + self.biases[-1]
        self.hidden_values.append(z)
        # Appling Softmax at last layer
        exp_y_hat = np.exp(z - np.max(z, axis=1, keepdims=True))
        data = exp_y_hat / np.sum(exp_y_hat, axis=1, keepdims=True)

        self.active_values.append(data)
        
        self.hidden_values = self.hidden_values
        self.active_values = self.active_values
        
        return self.hidden_values, self.active_values
    
    def compute_loss(self, y_pred, y_true, loss_type='cross_entropy', weight_decay=0):
        if loss_type == 'cross_entropy':
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        else:
            loss = 0.5 * np.mean(np.square(y_pred - y_true))
        
        if weight_decay > 0:
            reg_loss = sum(np.sum(np.square(W)) for W in self.weights)
            loss += 0.5 * weight_decay * reg_loss
        return loss
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == np.argmax(y_pred, axis=0))
    
    def predict(self, X):
        _, y_pred = self.feedforward(X)
        return np.argmax(y_pred[-1], axis=0)
    
    def backProp(self, X, y, loss_type="cross_entropy", weight_decay=0):
        m = X.shape[0]

        # compute ouput layer gradient
        if loss_type == "cross_entropy":
            da_k = self.active_values[-1] - y
        else:
            da_k = (self.active_values[-1] - y)*self.active_values*(1 - self.active_values)
        
        weights_grad = [None]*(len(self.weights))
        biases_grad = [None]*(len(self.weights))
        
        weights_grad[-1] = (1/m) * np.dot(self.active_values[-2].T, da_k)
        biases_grad[-1] = (1/m) * np.sum(da_k, axis=0, keepdims=True)

        if weight_decay > 0:
            weights_grad[-1] += weight_decay * self.weights[-1]

        # Gradients for Hidden Layer

        for i in range((len(self.weights)) - 2, -1, -1):
            da_k = np.dot(da_k, self.weights[i+1].T) * self.activation_derivative(self.active_values[i+1])
            weights_grad[i] = (1/m) * np.dot(self.active_values[i].T, da_k)
            biases_grad[i] = (1/m) * np.sum(da_k, axis=0, keepdims=True)

            if weight_decay > 0:
                weights_grad[i] += weight_decay * self.weights[i]
        
        self.history_grad.append((weights_grad, biases_grad))
        return weights_grad, biases_grad
    
    def train(self, X_train, y_train, X_val, y_val, optimizer, epochs=1,
              batch_size=32, loss_type='cross_entropy', beta=0.5, weight_decay=0, nag=False):
        
        if nag:
            prev_vw = [np.zeros_like(w) for w in self.weights]
            prev_vb = [np.zeros_like(w) for w in self.biases]
        
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []

        for _ in range(epochs):
            loss, acc = 0, 0
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                if nag:
                    for i in range(len(self.weights)):
                        self.weights[i] -= beta*prev_vw[i]
                        self.biases[i] -= beta*prev_vb[i]
                
                _, active_values = self.feedforward(X_batch)

                loss += self.compute_loss(active_values[-1], y_batch, loss_type, weight_decay)

                acc += self.accuracy(y_batch, active_values[-1])

                weights_grad, biases_grad = self.backProp(X_batch, y_batch, loss_type=loss_type)

                if nag:
                    self.weights, self.biases, prev_vw, prev_vb = optimizer.do_update(self.weights, self.biases, prev_vw, prev_vb, weights_grad, biases_grad)
                else:
                    self.weights, self.biases = optimizer.do_update(self.weights, self.biases, weights_grad, biases_grad)
            
            train_loss.append(loss/batch_size)
            train_acc.append(acc/batch_size)

            _, val_pred = self.feedforward(X_val)
            val_loss.append(self.compute_loss(val_pred[-1], y_val, loss_type, weight_decay))
            val_acc.append(self.accuracy(y_val, val_pred[-1]))

        return train_loss, val_loss, train_acc, val_acc
    
    def test(self, X_test, y_test, loss_type='cross_entropy', weight_decay=0):
        _, y_pred = self.feedforward(X_test)
        acc = self.accuracy(y_test, y_pred[-1])
        return acc

