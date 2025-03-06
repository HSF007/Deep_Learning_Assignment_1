import numpy as np


class NeuralNetwork:
    def __init__(self, input_features:int, hidden_layers:list[int], activation:str, output_features:int=10, weight_init:str="random"):
        self.input_features = input_features
        self.output_layer = output_features
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.weight_init = weight_init
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
        
    def feedforward(self, X):
        self.active_values = []
        self.hidden_values = []
        data = X

        for i, j in zip(self.weights, self.biases):
            z = i@data + j
            self.hidden_values.append(z)
            data = self.get_activation(self.activation)(z)
            self.active_values.append(data)
        
        return self.hidden_values, self.active_values
    
    def backProp(self, y):
        pass
