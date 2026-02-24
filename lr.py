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
