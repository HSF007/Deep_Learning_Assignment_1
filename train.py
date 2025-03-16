import os
import wandb
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


from Question_3 import *
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
                        "1,2,3".
                        THEY SHOULD BE COMMA SEPARATED VALUES ONLY''')
    parser.add_argument('-a', '--activation', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='ReLU')
    return parser.parse_args()


def preprocess_input(train, test):
    train = train.astype('float32') / 255
    test = test.astype('float32') / 255
    # Flatten the image
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return train, test


def onehot(x, class_num=10):
    return np.eye(class_num)[x]


def plot_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})


args = parse_arguments()

wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

# Load and preprocess data
if args.dataset == 'fashion_mnist':
    from tensorflow.keras.datasets import fashion_mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train, X_test = preprocess_input(X_train, X_test)
    y_train, y_test = onehot(y_train), onehot(y_test)
else:
    from tensorflow.keras.datasets import mnist #type:ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

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


def train(network, X_train, y_train, X_val, y_val, optimizer, epochs=1,
              batch_size=32, loss_type='cross_entropy', beta=0.5, weight_decay=0, nag=False):
        
        if nag:
            prev_vw = [np.zeros_like(w) for w in network.weights]
            prev_vb = [np.zeros_like(w) for w in network.biases]

        for epoch in range(epochs):
            num_batch = 0
            loss, acc = 0, 0
            for i in range(0, len(X_train), batch_size):
                num_batch += 1
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                if nag:
                    for i in range(len(network.weights)):
                        network.weights[i] -= beta*prev_vw[i]
                        network.biases[i] -= beta*prev_vb[i]
                
                _, active_values = network.feedforward(X_batch)

                loss += network.compute_loss(active_values[-1], y_batch, loss_type, weight_decay)

                acc += network.accuracy(y_batch, active_values[-1])

                weights_grad, biases_grad = network.backProp(X_batch, y_batch, loss_type=loss_type)

                if nag:
                    network.weights, network.biases, prev_vw, prev_vb = optimizer.do_update(network.weights, network.biases, prev_vw, prev_vb, weights_grad, biases_grad)
                else:
                    network.weights, network.biases = optimizer.do_update(network.weights, network.biases, weights_grad, biases_grad)
            

            _, val_pred = network.feedforward(X_val)
            val_loss = network.compute_loss(val_pred[-1], y_val, loss_type, weight_decay)
            val_acc = network.accuracy(y_val, val_pred[-1])

            loss = loss/num_batch
            acc = acc/num_batch

            wandb.log({
            'epoch': epoch + 1,
            'training_loss': loss/batch_size,
            'training_accuracy': acc/batch_size,
            'val_loss': val_loss,
            'val_accuracy': val_acc
            })


# Getting optimizer:
nag = False
if str(args.optimizer).lower() == 'sgd':
    optimizer = SGD(eta=float(args.learning_rate), weight_decay=float(args.weight_decay))
elif str(args.optimizer).lower() == 'momentum':
    optimizer = Momentum(eta=float(args.learning_rate), beta=float(args.momentum), weight_decay=float(args.weight_decay))
elif str(args.optimizer).lower() == 'nag':
    optimizer = NAG(eta=float(args.learning_rate), beta=float(args.momentum), weight_decay=float(args.weight_decay))
    nag = True
elif str(args.optimizer).lower() == 'rmsprop':
    optimizer = RMSProp(eta=float(args.learning_rate), beta=float(args.beta), eps=float(args.epsilon), weight_decay=float(args.weight_decay))
elif str(args.optimizer).lower() == 'adam':
    optimizer = Adam(eta=float(args.learning_rate), beta1=float(args.beta1), beta2=float(args.beta2), eps=float(args.epsilon), weight_decay=float(args.weight_decay))
elif str(args.optimizer).lower() == 'nadam':
    optimizer = Nadam(eta=float(args.learning_rate), beta1=float(args.beta1), beta2=float(args.beta2), eps=float(args.epsilon), weight_decay=float(args.weight_decay))
else:
    raise NameError(f'Error no optimizer such a {args.optimizer}.\nYou can choose optimizer from [sgd, momentum, nag, rmsprop, adam, nadam].')


train(
    network=network, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
    optimizer=optimizer, loss_type=str(args.loss), epochs=int(args.epochs),
    batch_size=int(args.batch_size), nag=(args.optimizer=='nag')
)

test_accuracy, test_pred = network.test(X_test, y_test)

wandb.log({
        "test_accuracy": test_accuracy
    })


classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
plot_confusion_matrix(y_test, test_pred[-1], classes)