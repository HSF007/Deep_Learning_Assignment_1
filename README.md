# Assignment-01

## Important Links
Weights and Biases Link: https://wandb.ai/da24m008-iit-madras/DL-Assignment-01/reports/DA6401-Assignment-1--VmlldzoxMTgzMjEzMw?accessToken=4c4tzzicb072x73hfj4gt25n5pcovzoh97xsmcx4go5zd1dzd5md3baxs2dbp5qy

GitHub Repositary Link: https://github.com/HSF007/Deep_Learning_Assignment_1

## Overview

This implements a neural network for classifying images from the Fashion-MNIST and MNIST datasets. The implementation includes:

- A flexible neural network with configurable hidden layers
- Activation functions (sigmoid, tanh, ReLU, identity)
- Various optimization (SGD, Momentum, NAG, RMSprop, Adam, NAdam)
- Weight initialization (random, Xavier)
- Loss functions (cross-entropy, mean squared error)
- Integration with Weights & Biases for experiment tracking and visualization

## Repository Structure

```
.
├── Question_1.py    # Loads and visualizes the sample from dataset
├── Question_2.py    # Neural network implementation with feed forward neural network and back propagation
├── Question_3.py    # All Optimization algorithms
├── Question_4.py    # Hyperparameter sweeps with wandb
├── train.py         # Main training script
└── README.md        # This file
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only for loading the datasets)
- Weights & Biases (wandb)

You can install the requirements using pip:

```bash
pip install numpy matplotlib tensorflow wandb
```

## Usage Instructions

### Basic Training

To train the model with default parameters:

```bash
python train.py --wandb_entity your_wandb_username --wandb_project your_project_name
```

### Command Line Arguments

The `train.py` script accepts the following command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | myprojectname | Project name in Weights & Biases |
| `-we`, `--wandb_entity` | myname | Your Weights & Biases username |
| `-d`, `--dataset` | fashion_mnist | Dataset to use: 'mnist' or 'fashion_mnist' |
| `-e`, `--epochs` | 10 | Number of training epochs |
| `-b`, `--batch_size` | 64 | Batch size for training |
| `-l`, `--loss` | cross_entropy | Loss function: 'mean_squared_error' or 'cross_entropy' |
| `-o`, `--optimizer` | nadam | Optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', or 'nadam' |
| `-lr`, `--learning_rate` | 0.001 | Learning rate for training |
| `-m`, `--momentum` | 0.5 | Momentum value for momentum and nag optimizers |
| `-beta`, `--beta` | 0.5 | Beta parameter for RMSprop |
| `-beta1`, `--beta1` | 0.5 | Beta1 parameter for Adam and NAdam |
| `-beta2`, `--beta2` | 0.5 | Beta2 parameter for Adam and NAdam |
| `-eps`, `--epsilon` | 1e-6 | Epsilon value for optimizers |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay (L2 regularization) |
| `-w_i`, `--weight_init` | Xavier | Weight initialization: 'random' or 'Xavier' |
| `-nhl`, `--num_layers` | 4 | Number of hidden layers |
| `-sz`, `--hidden_size` | 128 | Size of hidden layers (integer or comma-separated values) |
| `-a`, `--activation` | tanh | Activation function: 'identity', 'sigmoid', 'tanh', or 'ReLU' |

### Example Usage

Train a model with 3 hidden layers of size 64 using Adam optimizer:

```bash
python train.py --wandb_entity your_username --wandb_project your_project --num_layers 3 --hidden_size 64 --optimizer adam --learning_rate 0.001 --activation ReLU
```

Train a model with different sizes for hidden layers:
```bash
python train.py --wandb_entity your_username --wandb_project your_project --num_layers 3 --hidden_size 128,64,32 --optimizer nadam
```

### Running Hyperparameter Sweeps

To run hyperparameter sweeps using Weights & Biases:

```bash
python Question_4.py
```

This will run multiple training configurations according to the sweep configuration defined in `Question_4.py`.

## Neural Network Architecture

The implemented neural network is a fully connected feedforward network with the following features:

- Input layer: 784 neurons (28x28 pixels)
- Configurable hidden layers (number and size)
- Output layer: 10 neurons (one for each class) (As this assignment reqires train two dataset fashin-mnist and mnist)
- Activation functions (sigmoid, tanh, ReLU, identity)
- Softmax activation for the output layer (This will be the case for all runs as we need the probability of which class the data belong to.)

## Optimization Algorithms

The following optimization algorithms are implemented:

1. **Stochastic Gradient Descent (SGD)**
   - Basic gradient descent with configurable learning rate

2. **Momentum**
   - SGD with momentum to accelerate convergence

3. **Nesterov Accelerated Gradient (NAG)**
   - Improved version of momentum that looks ahead
   - Better convergence in some cases

4. **RMSprop**
   - Adaptive learning rate method
   - Divides the learning rate by the moving average of squared gradients

5. **Adam**
   - Combines ideas from momentum and RMSprop
   - Maintains adaptive learning rates for each parameter

6. **NAdam**
   - Nesterov-accelerated Adam
   - Combines NAG with Adam

## Results Visualization

All training metrics, including loss and accuracy, are logged to Weights & Biases. You can visualize:

- Training and validation loss vs epochs
- Training and validation accuracy vs epochs
- Confusion matrices
- Hyperparameter importance
- Model performance comparison

## Contributions
Me and myself only.