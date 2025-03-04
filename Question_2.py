import numpy as np


class Layer:
    def __init__(self, in_feature, out_feature, weight_init:str="random", bias:bool=True):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.weight_init = weight_init
    
    def parameters(self):
        self.biases = None
        if self.weight_init.lower() == "xavier":
            interval = np.sqrt(6/(self.in_feature + self.out_feature))
            self.weights = np.random.uniform(-interval, interval, size=(self.out_feature, self.in_feature))
            if self.bias:
                self.biases = np.random.uniform(-interval, interval, (self.out_feature, 1))
        else:
            self.weights = np.random.Generator.standard_normal(size=(self.out_feature, self.in_feature))
            if self.bias:
                self.biases = np.random.Generator.standard_normal(size=(self.out_feature, 1))
        return self.weights, self.biases
    
    def sigmoid(self, z, feed:bool=True):
        val = 1/(1 + np.exp(-z))
        if feed:
            return val
        else:
            return val*(1 - val)
    
    def identity(self, z, feed:bool=True):
        if feed:
            return z
        else:
            return np.ones(shape=z.shape)
    
    def tanh(self, z, feed:bool=True):
        val = np.tanh(z)
        if feed:
            return val
        else:
            return 1 - val*val
    
    def ReLu(self, z, feed:bool=True):
        if feed:
            return max(0, z)
        else:
            return 1 if z > 0 else 0



class NeuralNetwork:
    def __init__(self, input_features:int, hidden_layers:list, activations:list[str], output_layer:int=10, weight_init:str="random", bias:bool=True):
        self.input_features = input_features
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.bias = bias
        self.weight_init = weight_init
        self.activations = []
        self.weights = []
        self.biases = []
        for i in range(2+len(self.hidden_layers)):
            if not i:
                weight, bias = Layer(in_feature=self.input_features, out_feature=self.hidden_layers[0],
                                     weight_init=self.weight_init, bias=self.bias).parameters()
                self.weights.append(weight)
                self.biases.append(bias)
        
        for i in activations:
            if i.lower() == "identity":
                self.activations.append(Layer.identity())
            elif i.lower() == "sigmoid":
                self.activations.append(Layer.sigmoid())
            elif i.lower() == "tanh":
                self.activations.append(Layer.tanh())
            elif i.lower() == "relu":
                self.activations.append(Layer.ReLu())
            else:
                raise NameError(f"Error: No such activation as {i}.\nYou can choose activation fucntions from [identity, sigmoid, tanh, ReLU].")
        
    def feedforward(self, data):
        self.active_values = []
        self.hidden_values = []

        for i, j, a in zip(self.weights, self.biases, self.activations):
            z = i@data + j
            self.active_values.append(z)
            self.hidden_values.append(self.activations(z))
        
        self.y_hat = self.hidden_values.pop()
        return self.active_values, self.hidden_values, self.y_hat
    
    def backProp(self, X, y):
        pass
