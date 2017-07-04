"""
Предсказывание одной цифры из MNIST
"""

import numpy as np
import random as r

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.models import load_model

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = load_model("mnist_model.h5")

""" Пробуем предсказать """
our_images = X_train[r.randrange(0, 60000)]
from matplotlib import pyplot as plt
plt.imshow(our_images.reshape((28,28)))

#our_images = our_images.astype("float32")
result = np.ndarray.flatten(model.predict(our_images.reshape(1, 784)))
print('Prediction:')
print('This is 0 - ' + str(result[0]*100) + ' %')
print('This is 1 - ' + str(result[1]*100) + ' %')
print('This is 2 - ' + str(result[2]*100) + ' %')
print('This is 3 - ' + str(result[3]*100) + ' %')
print('This is 4 - ' + str(result[4]*100) + ' %')
print('This is 5 - ' + str(result[5]*100) + ' %')
print('This is 6 - ' + str(result[6]*100) + ' %')
print('This is 7 - ' + str(result[7]*100) + ' %')
print('This is 8 - ' + str(result[8]*100) + ' %')
print('This is 9 - ' + str(result[9]*100) + ' %')

#это выводит картинку на экран
plt.show()