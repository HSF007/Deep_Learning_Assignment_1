import os
import wandb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


from Question_3 import *
from Question_1 import get_sample_images
from Question_2 import NeuralNetwork

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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
    parser.add_argument('-sz', '--hidden_size', type=str, default=4, 
                        help='''You can pass an integer if all hidden layer have same neurons,
                        or you can pass a string with different neurons like:
                        "1,2,3" same as "1, 2, 3" both are accepted.
                        THEY SHOULD BE COMMA SEPARATED VALUES ONLY''')
    parser.add_argument('-a', '--activation', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='ReLU')
    return parser.parse_args()


args = parse_arguments()
    
    # Initialize wandb
# wandb.init(
#     project=args.wandb_project, 
#     entity=args.wandb_entity,
#     config=vars(args)
# )


def preprocess_input(train, test):
    train = train.astype('float32') / 255
    test = test.astype('float32') / 255
    # Flatten the image
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return train, test


def onehot(x, class_num=10):
    return np.eye(class_num)[x]

# Load and preprocess data
if args.dataset == 'fashion_mnist':
    from tensorflow.keras.datasets import fashion_mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Printing Sample images
    get_sample_images(X_train, y_train)

    X_train, X_test = preprocess_input(X_train, X_test)
    y_train, y_test = onehot(y_train), onehot(y_test)

    
else:
    from tensorflow.keras.datasets import mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Printing Sample images
    get_sample_images(X_train, y_train)

    X_train, X_test = preprocess_input(X_train, X_test)
    y_train, y_test = onehot(y_train), onehot(y_test)


val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]


hidden_layers_size = []
try:
    hidden_layers_size = [int(args.hidden_size)] * args.num_layers
except ValueError:
    hidden_size = str(args.hidden_size)
    for val in args.hidden_size.split(','):
        hidden_layers_size.append(int(val.strip()))
except Exception as error:
    print(f'''Error accured while adding hidden sizes from given input.
            Please check your input for hiddensize: {error}''')


# Initializing neural network
network = NeuralNetwork(
        input_features=784,
        hidden_layers=hidden_layers_size,
        activation=args.activation,
        output_features=10, 
        weight_init=args.weight_init
    )

# Getting optimizer:
nag = False
if str(args.optimizer).lower() == 'sgd':
    optimizer = SGD(eta=float(args.learning_rate))
elif str(args.optimizer).lower() == 'momentum':
    optimizer = Momentum(eta=float(args.learning_rate), beta=float(args.momentum))
elif str(args.optimizer).lower() == 'nag':
    optimizer = NAG(eta=float(args.learning_rate), beta=float(args.momentum))
    nag = True
elif str(args.optimizer).lower() == 'rmsprop':
    optimizer = RMSProp(eta=float(args.learning_rate), beta=float(args.beta), eps=float(args.epsilon))
elif str(args.optimizer).lower() == 'adam':
    optimizer = adam(eta=float(args.learning_rate), beta1=float(args.beta1), beta2=float(args.beta2), eps=float(args.epsilon))
elif str(args.optimizer).lower() == 'nadam':
    optimizer = Nadam(eta=float(args.learning_rate), beta1=float(args.beta1), beta2=float(args.beta2), eps=float(args.epsilon))
else:
    raise NameError(f'Error no optimizer such a {args.optimizer}.\nYou can choose optimizer from [sgd, momentum, nag, rmsprop, adam, nadam].')


# Training the model
training_loss, validation_loss, training_accuracy, validation_accuracy = network.train(
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    optimizer=optimizer, loss_type=str(args.loss), epochs=int(args.epochs),
    batch_size=int(args.batch_size), weight_decay=float(args.weight_decay),
    nag=nag
)

# Testing the model
test_accuracy = network.test(X_test, y_test, loss_type=str(args.loss))


epochs = np.arange(args.epochs) + 1
plt.plot(epochs, training_loss, label='training loss')
plt.plot(epochs, validation_loss, label='val loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss vs Epochs')
plt.show()


plt.plot(epochs, training_accuracy, label='training accuracy')
plt.plot(epochs, validation_accuracy, label='training accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.show()

print("Test Accuracy:", test_accuracy)