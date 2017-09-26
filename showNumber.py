

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

import threading


def showData(dataTesting_X):
    # showTesting 1
    plt.subplot(221)
    plt.imshow(dataTesting_X, cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()


# fix random seed for reproduceibility
seed = 7
numpy.random.seed(seed)

# load data
(dataTraning_X, dataTraning_y), (dataTesting_X, dataTesting_y) = mnist.load_data()

threading.Thread(target=showData(dataTesting_X[1])).start()