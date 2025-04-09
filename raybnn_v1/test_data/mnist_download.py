# digital canada may not download the mnist dataset
# you can download it on the local and upload the dataset into mnist folder

import numpy as np
from urllib import request
import gzip
import pickle

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]


def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def load_mnist_images(filename):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(filename):
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


download_mnist()
# Load the MNIST dataset
train_X = load_mnist_images("train-images-idx3-ubyte.gz")
train_y = load_mnist_labels("train-labels-idx1-ubyte.gz")
test_X = load_mnist_images("t10k-images-idx3-ubyte.gz")
test_y = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

train_X = train_X.reshape((train_X.shape[0], 28 * 28))
test_X = test_X.reshape((test_X.shape[0], 28 * 28))

train_y = np.eye(10)[train_y]
test_y = np.eye(10)[test_y]

# Normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# Use binary writes to ensure consistent line breaks on linux and windows
def save_array(filename, array):
    with open(filename, "wb") as f:
        np.savetxt(f, array, fmt="%.8f", delimiter=",", newline="\n")


# save .dat
save_array("./mnist/mnist_train_X.dat", train_X)
save_array("./mnist/mnist_train_y.dat", train_y)
save_array("./mnist/mnist_test_X.dat", test_X)
save_array("./mnist/mnist_test_y.dat", test_y)
