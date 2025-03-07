import numpy as np


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
        active_values = []
        hidden_values = []
        data = X

        for index, param in enumerate(zip(self.weights, self.biases)):
            z = param[0]@data + param[1]
            hidden_values.append(z)
            if index != len(self.weights) - 1:
                data = self.activation(z)
            else:
                # Appling Softmax at last layer
                exp_y_hat = np.exp(z - np.max(z))
                data = exp_y_hat / np.sum(exp_y_hat)
            
            active_values.append(data)
        
        
        return hidden_values, active_values
    
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
    
    def backProp(self, X, y, active_value, hidden_values, loss="cross_entropy"):
        m = X.shape[0]
        y_one_hot = np.eye(self.output_features)[y].T

        # compute ouput layer gradient
        grad_ak_L_theta = active_value[-1] - y_one_hot
        
        gradients = [grad_ak_L_theta]
        # Gradients for Hidden Layer
        for i in reversed(range(len(self.weights) - 1)):
            dW = np.dot(gradients[-1], active_value[i+1].T) / m
            db = np.mean(gradients[-1], axis=1, keepdims=True)
            
            # If not output layer, apply activation derivative
            if i > 0:
                dZ = np.dot(self.weights[i+1].T, gradients[-1]) * self.activation_derivative(active_value[i+1])
            else:
                break
            
            gradients.append(dZ)

        pass
