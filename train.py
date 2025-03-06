import os
import wandb
import argparse
import numpy as np
import pandas as pd
from Question_1 import get_sample_images
from Question_2 import NeuralNetwork

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Network Training Script')
    
    # All arguments as specified in the assignment
    parser.add_argument('-wp', '--wandb_project', default='myprojectname', 
                        help='Project name for Weights & Biases')
    parser.add_argument('-we', '--wandb_entity', default='myname', 
                        help='Wandb Entity')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], 
                        default='fashion_mnist', help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=1, 
                        help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=4, 
                        help='Batch size for training')
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], 
                        default='cross_entropy', help='Loss function')
    parser.add_argument('-o', '--optimizer', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], 
                        default='sgd', help='Optimization algorithm')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, 
                        help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, 
                        help='Momentum for momentum and NAG optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.5, 
                        help='Beta for RMSprop')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, 
                        help='Beta1 for Adam and Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, 
                        help='Beta2 for Adam and Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, 
                        help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, 
                        help='Weight decay for regularization')
    parser.add_argument('-w_i', '--weight_init', choices=['random', 'Xavier'], 
                        default='random', help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, 
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, 
                        help='Number of neurons in hidden layers')
    parser.add_argument('-a', '--activation', 
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'], 
                        default='ReLU', help='Activation function')
    



args = parse_arguments()
    
    # Initialize wandb
wandb.init(
    project=args.wandb_project, 
    entity=args.wandb_entity,
    config=vars(args)
)

# Load and preprocess data
if args.dataset == 'fashion_mnist':
    from tensorflow.keras.datasets import fashion_mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
else:
    from tensorflow.keras.datasets import mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

get_sample_images(X_train, y_train)

val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]

network = NeuralNetwork(
        input_size=784, 
        output_size=10, 
        hidden_layers=[args.hidden_size] * args.num_layers,
        activation=args.activation,
        weight_init=args.weight_init
    )