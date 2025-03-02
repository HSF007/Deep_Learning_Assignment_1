import os
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist


classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()