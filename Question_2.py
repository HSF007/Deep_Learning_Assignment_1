import os
import numpy as np
import matplotlib.pyplot as plt





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




class Layer:
    def __init__(self, value_in, value_out, bias:bool=True):
        self.value_in = value_in
        self.value_out = value_out
        self.bias = bias
        self.weights = np.random.Generator.standard_normal(size=(value_out, value_in))

class Network:
    def __init__(self, )