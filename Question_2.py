import os
import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, in_feature, out_feature, weight_init:str="random", bias:bool=True):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = bias
        self.weight_init = weight_init
    
    def parameters(self):
        self.biases = None
        if self.weight_init == "Xavier":
            interval = np.sqrt(6/(self.feature_in + self.feature_out))
            self.weights = np.random.uniform(-interval, interval, size=(self.in_feature, self.out_feature))
            if self.bias:
                 self.biases = np.random.uniform(-interval, interval, (1, self.out_feature))
        else:
            self.weights = np.random.Generator.standard_normal(size=(self.in_feature, self.out_feature))
            if self.bias:
                 self.biases = np.random.Generator.standard_normal(size=(1, self.out_feature))
        return self.weights, self.biases


class BuildNetwork:
    def __init__(self, network:list, num_layers:int=1):
        self.network = network
        self.num_layers = num_layers
    
    def all_weights_biases(self):
        self.weights_biases = []
        self.activation = []
        biases = []
        for layer in self.network:
            if type(layer) == Layer:
                w, b = layer.parameters()
                self.weights_biases.append(w)
                biases.append(b)
        
        self.weights_biases.extend(biases)
    
    def forwardFeed(self, input):
        self.all_weights_biases()

        pass




class Network:
    def __init__(self, nodes:list, activations:list, bias:bool=True):
        """
        Attributes
        ----------
        nodes : list
            list of number of nodes in each layer
            Example: A network contains input=100, hidden_1=200, hidden_2=200 and output=10
            then nodes = [100, 200, 200, 10]
        
        activations : list
            List of activation functions for hidden and output layer.
        
        bias : bool
            True if network contains bias else False
        """

        self.nodes = nodes
        self.activations = activations
        self.bias = bias
    
    def layer(self, in_feature:int, out_feature:int, bias:bool=True):
        biases = None
        interval = np.sqrt(6 / (in_feature + out_feature))
        weights = np.random.uniform(-interval, interval, (in_feature, out_feature))
        if bias:
            biases = np.random.uniform(-interval, interval, (1, out_feature))
        return weights, biases
    


    def parameters(self):
        """Initializes weights and biases."""

        self.weights_biases = []
        for i in range(len(self.nodes) - 1):
            shape = (self.nodes[i], self.nodes[i+1])
            if i and self.activations[i-1]=="relu":
                std = np.sqrt(2 / shape[0])
                self.weights_biases.append(np.random.normal(0, std, shape))
                if self.bias:
                    self.weights_biases.append(np.random.normal(0, std, (1, shape[1])))
            
            elif i and (self.activations[i-1] == "sigmoid" or self.activations[i-1] == "tanh"):
                interval = np.sqrt(6 / (shape[0] + shape[1]))
                self.weights_biases.append(np.random.uniform(-interval, interval, shape))
                if self.bias:
                    self.weights_biases.append(np.random.uniform(-interval, interval, (1, shape[1])))
            
            else:
                self.weights_biases.append(np.random.uniform(-0.01, 0.01, shape))
                if self.bias:
                    self.weights_biases.append(np.random.uniform(-0.01, 0.01, (1, shape[1])))


    def forwardFeed(self):
        pass


class Layer:
    def __init__(self, activation, hidden_size:int, weight_init:str="random", bias:bool=True)