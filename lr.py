import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy

from PIL import Image
from scipy import ndimage
from utils import load_dataset

training_set_x_orig, training_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# plt.imshow(training_set_x_orig[100])
# plt.show()

# number of training examples and test examples
m_training = training_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]

# flatten the images
training_set_x_orig_flatten = training_set_x_orig.reshape(
    m_training, -1).T  # shape : (64*64,m_train)
test_set_x_orig_flatten = test_set_x_orig.reshape(
    m_test, -1).T  # shape: (64*64,m_test)

# normalization, this is one way .. there is another
train_x = training_set_x_orig_flatten / 255
test_x = test_set_x_orig_flatten / 255

print(train_x.shape[0])


def sigmoid(z):
    f = 1 / (1+np.exp(-z))
    return f


def init_params(dim):

    w = np.zeros((dim, 1))
    b = .0

    return w, b


def propagation(w, b, X, Y):

    # n - examples
    m = X.shape[1]

   # forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))

    # backward propagation
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum((A-Y))

    cost = np.squeeze(cost)  # to ensure cost is a np array

    gradients = {"dw": dw,
                 "db": db}

    return gradients, cost


def model(n_x):
    n_x = train_x.shape[0]
    print(n_x)
