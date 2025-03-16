import wandb
import os
from Question_1 import get_sample_images
from Question_2 import *
from tensorflow.keras.datasets import fashion_mnist #type:ignore

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sweep_config = {
    'method': 'random',  # Can also use 'random' or 'grid' or 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'Xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'ReLU']}
    }
}


def preprocess_input(train, test):
    train = train.astype('float32') / 255
    test = test.astype('float32') / 255
    # Flatten the image
    train = train.reshape(train.shape[0], -1)
    test = test.reshape(test.shape[0], -1)
    return train, test


def onehot(x, class_num=10):
    return np.eye(class_num)[x]


def get_optimizer(config):
    optimizer = None
    if str(config.optimizer).lower() == 'sgd':
        optimizer = SGD(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
    elif str(config.optimizer).lower() == 'momentum':
        optimizer = Momentum(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
    elif str(config.optimizer).lower() == 'nag':
        optimizer = NAG(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
        nag = True
    elif str(config.optimizer).lower() == 'rmsprop':
        optimizer = RMSProp(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
    elif str(config.optimizer).lower() == 'adam':
        optimizer = Adam(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
    elif str(config.optimizer).lower() == 'nadam':
        optimizer = Nadam(eta=float(config.learning_rate), weight_decay=float(config.weight_decay))
    else:
        raise NameError(f'Error no optimizer such a {config.optimizer}.\nYou can choose optimizer from [sgd, momentum, nag, rmsprop, adam, nadam].')
    return optimizer



def train(network, X_train, y_train, X_val, y_val, optimizer, epochs=1,
              batch_size=32, loss_type='cross_entropy', beta=0.5, nag=False):
        
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

                loss += network.compute_loss(active_values[-1], y_batch, loss_type)

                acc += network.accuracy(y_batch, active_values[-1])

                weights_grad, biases_grad = network.backProp(X_batch, y_batch, loss_type=loss_type)

                if nag:
                    network.weights, network.biases, prev_vw, prev_vb = optimizer.do_update(network.weights, network.biases, prev_vw, prev_vb, weights_grad, biases_grad)
                else:
                    network.weights, network.biases = optimizer.do_update(network.weights, network.biases, weights_grad, biases_grad)
            

            _, val_pred = network.feedforward(X_val)
            val_loss = network.compute_loss(val_pred[-1], y_val, loss_type)
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


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = preprocess_input(X_train, X_test)
y_train, y_test = onehot(y_train), onehot(y_test)

val_split = int(0.9 * len(X_train))
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]


def trainer(config):
    hidden_layers = [config.hidden_size]*config.num_layers

    # Initialize the model
    network = NeuralNetwork(
        input_features=784,
        hidden_layers=hidden_layers,
        activation=config.activation,
        output_features=10, 
        weight_init=config.weight_init
    )

    optimizer = get_optimizer(config)

    train(
        network, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        optimizer=optimizer, loss_type='cross_entropy', epochs=config.epochs,
        batch_size=config.batch_size, nag=(config.optimizer == 'nag')
    )

    test_accuracy, _ = network.test(X_test, y_test)

    return test_accuracy



def sweep_train():
    wandb.init()
    config = wandb.config

    run_name = f"hl_{config.num_layers}_sz_{config.hidden_size}_bs_{config.batch_size}_ac_{config.activation}"
    wandb.run.name = run_name

    test_accuracy = trainer(config)
    
    # Train and evaluate
    wandb.log({
        "test_accuracy": test_accuracy
    })


if __name__ == "__main__":
    # Run the sweep
    sweep_id = wandb.sweep(sweep_config, project='DL-Assignment-01')

    wandb.agent(sweep_id, sweep_train, count=120)  # Run 120 different configurations
    sweep_train()
