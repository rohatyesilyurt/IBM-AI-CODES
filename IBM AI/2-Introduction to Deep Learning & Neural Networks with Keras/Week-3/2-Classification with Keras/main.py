import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# import the data
from keras.datasets import mnist

# read the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# plt.imshow(x_train[0])
# print(y_train[0])
# plt.show()

# # flatten images into one-dimensional vector
num_pixels = x_train.shape[1] * x_train.shape[2] # find size of one-dimensional vector

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') # flatten training images
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]
print(y_train.shape)
print(num_classes)

# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))   

model.save('classification_model.h5')