import numpy as np
import matplotlib.pyplot as plt
import wandb
import argparse

# Getting arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Learning Assignment-01')
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname')
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='fashion_mnist')
    return parser.parse_args()


# Ploting Sample images and recording logs in WandB
def get_sample_images(X_train, y_train, classes):

    fig, axs = plt.subplots(2,5)

    for i, ax in zip(range(10), axs.flat):
        ax.axis('off')
        ax.imshow(X_train[np.where(y_train == i)[0][0]], cmap='gray')
        ax.set_title(classes[i])

    plt.tight_layout()
    plt.savefig('sample_images.png')
    wandb.log({'sample_images': wandb.Image('sample_images.png')})


if __name__ == "__main__":
    # Taking arguments for run
    args = parse_arguments()

    # Initializing WnadB
    wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
    
    if args.dataset == 'fashion_mnist':
        from tensorflow.keras.datasets import fashion_mnist #type:ignore
        (X_train, y_train), _ = fashion_mnist.load_data()

        classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        get_sample_images(X_train, y_train, classes)

    else:
        from tensorflow.keras.datasets import mnist #type:ignore
        (X_train, y_train), _ = mnist.load_data()

        classes = [i for i in range(0, 10)]
        get_sample_images(X_train, y_train, classes)