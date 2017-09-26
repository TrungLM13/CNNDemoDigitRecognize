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


def baseline_module():
    #create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fix random seed for reproduceibility
seed = 7
numpy.random.seed(seed)

# load data
(dataTraning_X, dataTraning_y), (dataTesting_X, dataTesting_y) = mnist.load_data()

#showTesting 1
plt.subplot(221)
plt.imshow(dataTraning_X[1], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#reshape
dataTraning_X = dataTraning_X.reshape(dataTraning_X.shape[0], 1, 28, 28).astype('float32')
dataTesting_X = dataTesting_X.reshape(dataTesting_X.shape[0], 1, 28, 28).astype('float32')

#normalize inputs from 0-255 to 0-1
dataTraning_X = dataTraning_X / 255
dataTesting_X = dataTesting_X / 255

#one hot encode outputs
dataTraning_y = np_utils.to_categorical(dataTraning_y)
dataTesting_y = np_utils.to_categorical(dataTesting_y)

num_classes = dataTesting_y.shape[1]

#build model
model = baseline_module()

#fit model
model.fit(dataTraning_X, dataTraning_y, validation_data=(dataTesting_X, dataTesting_y), epochs=10, batch_size=200,verbose=2)

#final evaluation of model
scores = model.evaluate(dataTesting_X, dataTesting_y, verbose=0)

classifyLayer = model.predict(dataTesting_X[1], 256, 0)


print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("Number of classify: %d" % classifyLayer)