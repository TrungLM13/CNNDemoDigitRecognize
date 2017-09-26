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
from scipy.misc import imread
K.set_image_dim_ordering('th')

from keras.models import load_model

import threading


def baseline_module(num_classes):
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def showData(dataTesting_X):
    # showTesting 1
    plt.subplot(221)
    plt.imshow(dataTesting_X[0], cmap=plt.get_cmap('gray'))
    # show the plot
    plt.show()


def machineLearning(dataTraning_X, dataTraning_y, dataTesting_X, dataTesting_y):

    model = None

    try:
        with open('cnn_digit_number.h5') as f:
            model = load_model('cnn_digit_number.h5')

    except IOError as e:
        print 'Trouble opening file'

        dataTraning_X = dataTraning_X.reshape(dataTraning_X.shape[0], 1, 28, 28).astype('float32')
        dataTesting_X = dataTesting_X.reshape(dataTesting_X.shape[0], 1, 28, 28).astype('float32')

        # normalize inputs from 0-255 to 0-1
        dataTraning_X = dataTraning_X / 255
        dataTesting_X = dataTesting_X / 255

        # one hot encode outputs
        dataTraning_y = np_utils.to_categorical(dataTraning_y)
        dataTesting_y = np_utils.to_categorical(dataTesting_y)

        num_classes = dataTesting_y.shape[1]

        # build model
        model = baseline_module(num_classes)

        # fit model
        model.fit(dataTraning_X, dataTraning_y, validation_data=(dataTesting_X, dataTesting_y), epochs=10, batch_size=200, verbose=2)

        # final evaluation of model
        scores = model.evaluate(dataTesting_X, dataTesting_y, verbose=0)

        model.save('cnn_digit_number.h5')

    im = imread("so4.png")

    im = im / 255
    im = np_utils.to_categorical(im)

    pr = model.predict_classes(im.reshape(1, 1, 28, 28))
    print(pr)
    # print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


# fix random seed for reproduceibility
seed = 7
numpy.random.seed(seed)

# load data
(dataTraning_X, dataTraning_y), (dataTesting_X, dataTesting_y) = mnist.load_data()

threading.Thread(target=machineLearning(dataTraning_X, dataTraning_y, dataTesting_X, dataTesting_y))