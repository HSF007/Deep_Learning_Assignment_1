import os
import wandb
import argparse
import numpy as np
import pandas as pd
from Question_1 import get_sample_images
from Question_2 import NeuralNetwork

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def datatype_check(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Learning Assignment-01')
    parser.add_argument('-wp', '--wandb_project', default='myprojectname')
    parser.add_argument('-we', '--wandb_entity', default='myname')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'fashion_mnist'], default='fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=1,)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-l', '--loss', choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy')
    parser.add_argument('-o', '--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-beta', '--beta', type=float, default=0.5)
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5)
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6)
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_i', '--weight_init', choices=['random', 'Xavier'], default='random')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1)
    parser.add_argument('-sz', '--hidden_size', type=datatype_check, default=4, 
                        help='''You can pass an integer if all hidden layer have same neurons,
                        or you can pass a string with different neurons like:
                        "1,2,3" same as "1, 2, 3" both are accepted.
                        THEY SHOULD BE COMMA SEPARATED VALUES ONLY''')
    parser.add_argument('-a', '--activation', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='ReLU')
    return parser.parse_args()


args = parse_arguments()
    
    # Initialize wandb
wandb.init(
    project=args.wandb_project, 
    entity=args.wandb_entity,
    config=vars(args)
)


def preprocess_input(train, test):
    train = train.astype('float32') / 255.0
    test = test.astype('float32') / 255.0
    # Flatten the image
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return train, test


# Load and preprocess data
if args.dataset == 'fashion_mnist':
    from tensorflow.keras.datasets import fashion_mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Printing Sample images
    get_sample_images(X_train, y_train)

    X_train, X_test = preprocess_input(X_train, X_test)
    
else:
    from tensorflow.keras.datasets import mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Printing Sample images
    get_sample_images(X_train, y_train)

    X_train, X_test = preprocess_input(X_train, X_test)


val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]


if type(args.hidden_size) == int:
    hidden_layers_size = [args.hidden_size] * args.num_layers
else:
    hidden_layers_size = []
    for val in args.hidden_size.split():
        try:
            hidden_layers_size.append(int(val.strip()))
        except Exception as error:
            print(f'''Error accured while adding hidden sizes from given input.
                  Please check your input for hiddensize: {error}''')


network = NeuralNetwork(
        input_size=784, 
        output_size=10, 
        hidden_layers=hidden_layers_size,
        activation=args.activation,
        weight_init=args.weight_init
    )