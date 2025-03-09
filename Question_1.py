import numpy as np
import matplotlib.pyplot as plt
import wandb



def get_sample_images(X_train, y_train):
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    fig, axs = plt.subplots(2,5)

    for i, ax in zip(range(10), axs.flat):
        ax.axis('off')
        ax.imshow(X_train[np.where(y_train == i)[0][0]], cmap='gray')
        ax.set_title(classes[i])

    plt.tight_layout()
    # plt.show()
    # plt.savefig('sample_images.png')
    # wandb.log({'sample_images': wandb.Image('sample_images.png')})