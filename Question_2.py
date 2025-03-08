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
                w = np.random.uniform(-interval, interval, size=(layer_sizes[i+1], layer_sizes[i]))
                b = np.random.uniform(-interval, interval, (layer_sizes[i+1], 1))
            else:
                w = np.random.Generator.standard_normal(size=(layer_sizes[i+1], layer_sizes[i]))
                b = np.random.Generator.standard_normal(size=(layer_sizes[i+1], 1))
            
            self.weights.append(w)
            self.biases.append(b)
        
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
        self.active_values = [X.T]
        self.hidden_values = []

        for index, param in enumerate(zip(self.weights, self.biases)):
            z = param[0]@self.active_values[-1] + param[1]
            self.hidden_values.append(z)
            if index != len(self.weights) - 1:
                data = self.activation(z)
            else:
                # Appling Softmax at last layer
                exp_y_hat = np.exp(z - np.max(z))
                data = exp_y_hat / np.sum(exp_y_hat)
            
            self.active_values.append(data)
        
        
        return self.hidden_values, self.active_values
    
    def compute_loss(self, y_pred, y_true, loss_type='cross_entropy', weight_decay=0):
        if loss_type == 'cross_entropy':
            epsilon = 1e-15 # To avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y_true * np.log(y_pred)).mean()
        else:
            loss = 0.5 * np.mean(np.square(y_pred - y_true))
        
        if weight_decay > 0:
            reg_loss = sum(np.sum(np.square(W)) for W in self.weights)
            loss += 0.5 * weight_decay * reg_loss
        
        return loss
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == np.argmax(y_pred, axis=0))
    
    def backProp(self, X, y, loss="cross_entropy"):
        m = X.shape[0]
        y_one_hot = np.eye(self.output_features)[y].T

        # compute ouput layer gradient
        if loss == "cross_entropy":
            da_k = self.active_values[-1] - y_one_hot
        else:
            pass
        
        weights_grad = []
        biases_grad = []

        # Gradients for Hidden Layer
        for i in reversed(range(len(self.weights))):
            dW = np.dot(da_k, self.active_values[i].T)/m
            db = np.mean(da_k, axis=1, keepdims=True)

            dh_k = self.weights.T @ da_k
            da_k = np.multiply(dh_k, self.activation_derivative(self.hidden_values[i]))
            weights_grad.append(dW)
            biases_grad.append(db)
        self.history_grad.append((weights_grad, biases_grad))
        return weights_grad, biases_grad
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=32,
              learning_rate=0.1, optimizer="sgd", loss_type='cross_entropy', weight_decay=0, beta=0.5):
        if optimizer.lower() == "sgd":
            opt = SGD(learning_rate)
        
        elif optimizer.lower() == 'momentum':
            opt = Momentum(learning_rate, beta)
        
        elif optimizer.lower() == 'nag':
            opt = NAG(learning_rate, beta)
            prev_vw = np.zeros_like(self.weights)
            prev_vb = np.zeros_like(self.biases)
        
        elif optimizer.lower() == 'rmsprop':
            opt = RMSProp()
        
        elif optimizer.lower() == 'adam':
            opt = adam(learning_rate)
        
        elif optimizer.lower() == 'nadam':
            opt = Nadam(learning_rate)
        
        else:
            raise NameError(f"Error: No such optimizer as {optimizer}.\nYou can choose optimizer from [sgd, momentum, nag, rmsprop, adam, nadam].")
        
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []

        for _ in range(epochs):
            loss, acc = 0, 0
            for i in range(0, X_train, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                if optimizer.lower() == 'nag':
                    self.weights -= beta*prev_vw
                    self.biases -= beta*prev_vb
                
                _, active_values = self.feedforward(X_batch)

                loss += self.compute_loss(active_values[-1], y_batch, loss_type, weight_decay)

                accu += self.accuracy(y_batch, active_values[-1])

                weights_grad, biases_grad = self.backProp(X_batch, y_batch, loss_type=loss_type)

                if optimizer.lower() == 'nag':
                    self.weights, self.biases, prev_vw, prev_vb = opt.do_update(self.weights, self.biases, prev_vw, prev_vb, weights_grad, biases_grad)
                else:
                    self.weights, self.biases = opt.do_update(self.weights, self.biases, weights_grad, biases_grad)
            
            train_loss.append(loss/batch_size)
            train_acc.append(acc/batch_size)

            _, val_pred = self.feedforward(X_val)
            val_loss.append(self.compute_loss(val_pred[-1], y_val, loss_type, weight_decay))
            val_acc.append(self.accuracy(y_val, val_pred[-1]))
        
        wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            })

