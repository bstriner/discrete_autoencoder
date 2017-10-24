import numpy as np
from keras.datasets import mnist


def mnist_clean(x):
    return np.reshape(np.array(x).astype(np.float32) / 255.0, (x.shape[0], -1))


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_clean(xtrain), mnist_clean(xtest)
