import numpy as np
from Question_3 import *

class NeuralNetwork:
    def __init__(self, input_features:int, hidden_layers:list[int], activation:str, output_features:int=10, weight_init:str="random"):
        # Taking all parameters
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_layers = hidden_layers
        self.weight_init = weight_init

        # Initilizing activation function and it's derivative
        self.activation = self.get_activation(activation)
        self.activation_derivative = self.get_activation_derivative(activation)

        # Creating list to store weights and biases
        self.weights = []
        self.biases = []

        # Creating network sructure
        layer_sizes = [input_features] + hidden_layers + [output_features]

        # Initilizing weights for all layers and stroting them
        for i in range(len(layer_sizes) - 1):
            if self.weight_init.lower() == "xavier":
                interval = np.sqrt(6/((layer_sizes[i] + layer_sizes[i+1])))
                w = np.random.uniform(-interval, interval, size=(layer_sizes[i], layer_sizes[i+1]))
                b = np.random.uniform(-interval, interval, (1, layer_sizes[i+1]))
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
                b = np.random.randn(1, layer_sizes[i+1]) * 0.01
            
            self.weights.append(w)
            self.biases.append(b)
        
        self.weights = self.weights
        self.biases = self.biases
        
    # function to get activation function
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
    
    # Function to get derivative of activation fucntion
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
    
    # Forward feed
    def feedforward(self, X):
        self.active_values = [X]
        self.hidden_values = []

        # Calculating active and hidden values and storing them
        for i in range(len(self.weights) - 1):
            z = np.dot(self.active_values[-1], self.weights[i]) + self.biases[i]
            self.hidden_values.append(z)
            data = self.activation(z)
            self.active_values.append(data)

        z = np.dot(self.active_values[-1], self.weights[-1]) + self.biases[-1]
        self.hidden_values.append(z)

        # Appling Softmax at last layer
        exp_y_hat = np.exp(z - np.max(z, axis=1, keepdims=True))
        data = exp_y_hat / (np.sum(exp_y_hat, axis=1, keepdims=True) + 1e-15)

        self.active_values.append(data)
        
        self.hidden_values = self.hidden_values
        self.active_values = self.active_values
        
        return self.hidden_values, self.active_values
    
    # Computin Loss
    def compute_loss(self, y_pred, y_true, loss_type='cross_entropy'):
        if loss_type == 'cross_entropy':
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))
        else:
            loss = 0.5 * np.mean(np.square(y_pred - y_true))
        return loss
    
    # Computing accuracy
    def accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_true ,axis=1) == np.argmax(y_pred, axis=1))
    
    def predict(self, X):
        _, y_pred = self.feedforward(X)
        return np.argmax(y_pred[-1], axis=0)
    
    # Back Propagation
    def backProp(self, X, y, loss_type="cross_entropy", weight_decay=0):
        m = X.shape[0]

        # compute ouput layer gradient
        if loss_type == "cross_entropy":
            da_k = self.active_values[-1] - y
        else:
            da_k = (self.active_values[-1] - y)*self.active_values[-1]*(1 - self.active_values[-1])
        
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
        
        return weights_grad, biases_grad
    
    def test(self, X_test, y_test):
        _, y_pred = self.feedforward(X_test)
        acc = self.accuracy(y_test, y_pred[-1])
        return acc, y_pred

