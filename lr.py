import copy
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
    cost = -1/m * (np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))

    # backward propagation
    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum((A-Y))

    cost = np.squeeze(cost)  # to ensure cost is a np array

    gradients = {"dw": dw,
                 "db": db}

    return gradients, cost


def optimize(w, b, X, Y, learning_rate=0.009, num_iterations=100, print_cost=False):

    costs = []
    for i in range(num_iterations):
        gradients, cost = propagation(w, b, X, Y)
        w = w - learning_rate*gradients["dw"]
        b = b - learning_rate*gradients["db"]

        if print_cost:
            if i % 100 == 0:
                costs.append(cost)
                print(
                    f"Iteration : {i} \tCost : {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": gradients["dw"], "db": gradients["db"]}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    predictions = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)) + b

    for i in range(A.shape[1]):
        if (A[0, i] > 0.5):
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0

    return predictions


def model(X_train, Y_train, X_test, Y_test, num_its=2000, learning_rate=0.5, print_cost=False):
    w, b = init_params(X_train.shape[0])
    params, grads, costs = optimize(
        w, b, X_train, Y_train, learning_rate=learning_rate, num_iterations=num_its, print_cost=print_cost)

    w = params["w"]
    b = params["b"]

    predictions_train_set = predict(w, b, X_train)
    predictions_test_set = predict(w, b, X_test)

    info = {
        "costs": costs,
        "prediction on training set": predictions_train_set,
        "prediction on test set": predictions_test_set,
        "w": w,
        "b": b,
        "alpha": learning_rate,
        "n. iterations": num_its
    }

    return info


lr_model = model(train_x, training_set_y, test_x,
                 test_set_y, 2000, 0.005, True)
