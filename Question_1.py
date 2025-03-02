import os
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist


classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

fig, axs = plt.subplots(2,5)

row = 0
for i, ax in zip(range(10), axs.flat):
    ax.axis('off')
    ax.imshow(X_train[np.where(y_train == i)[0][0]], cmap='gray')
    ax.set_title(classes[i])

plt.show()