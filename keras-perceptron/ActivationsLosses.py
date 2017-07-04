"""
Эта программа основывается на "example1.py" из этого же проекта. Здесь подставляются различные функции
активации и ошибки. В конце файла есть таблица с результатами.

"""

import numpy as np

from keras import losses
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense

import datetime

""" 1)  Загрузка обучающих и тестовых примеров из MNIST.
        Обратите внимание на приведение значений яркостей из диапазона 0..255 в [0,1]:
            X_train /= 255
            X_test  /= 255 
"""
nb_classes = 10

activations = ("softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear")

losses = ("mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_logarithmic_error", "squared_hinge", "hinge", "logcosh", "categorical_crossentropy", "poisson", "cosine_proximity")

now = datetime.datetime.now()
indicator = str(now.year) + str(now.month) + str(now.day) + "_" + str(now.hour) + "-" + str(now.minute)
output_file_name = "data/ActivationsLosses.txt"

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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


for act in activations:
    for loss in losses:
        print("Activation: " + act + "; Loss: " + loss)
        model = Sequential()
        model.add(Dense(input_dim=784, activation=act, units=100))
        model.add(Dense(units=nb_classes, input_dim=784, activation='softmax'))

        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=2, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)

        #Заменим десятичную точку на запятую, чтоб потом не морочиться с Excel'ем
        line = act + '\t' + loss + '\t' + str(score[1]).replace('.', ',') + '\n'
        f = open(output_file_name, 'a')
        f.write(line)
        f.close()

# Сохраним модель
#model.save("my_model.h5")

""" РЕЗУЛЬТАТЫ

о Доступные функции активации   https://keras.io/activations/
о Доступные функции потерь      https://keras.io/losses/
о Доступные оптимизаторы        https://keras.io/optimizers/

Однослойная модель
=========================================================================================
 №  |    ФА1         |   ФП                             |  Оптимизатор  |   Точность    |
=========================================================================================
1.1 |    softmax     |  categorical_crossentropy        |    adam       |   0.9279      |
-----------------------------------------------------------------------------------------
1.2 |    relu        |  categorical_crossentropy        |    adam       |   0.0980      | :(
-----------------------------------------------------------------------------------------
1.3 |    elu         |  categorical_crossentropy        |    adam       |   0.0980      | :(
-----------------------------------------------------------------------------------------
1.4 |    softplus    |  categorical_crossentropy        |    adam       |   0.9289      | :D
-----------------------------------------------------------------------------------------
1.5 |    softsign    |  categorical_crossentropy        |    adam       |   0.1603      | :(
-----------------------------------------------------------------------------------------
1.6 |    tanh        |  categorical_crossentropy        |    adam       |   0.1551      | :(
-----------------------------------------------------------------------------------------
1.7 |    sigmoid     |  categorical_crossentropy        |    adam       |   0.9274      |
-----------------------------------------------------------------------------------------
1.8 |  hard_sigmoid  |  categorical_crossentropy        |    adam       |   0.8898      |
----------------------------------------------------------------------------------------- 
1.9 |    linear      |  categorical_crossentropy        |    adam       |   0.1420      | :(
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
2.1 |    softmax     |  logcosh                         |    adam       |   0.9322      | :D
-----------------------------------------------------------------------------------------
2.2 |    relu        |  logcosh                         |    adam       |   0.8339      |
-----------------------------------------------------------------------------------------
2.3 |    elu         |  logcosh                         |    adam       |   0.8623      |
-----------------------------------------------------------------------------------------
2.4 |    softplus    |  logcosh                         |    adam       |   0.9134      |
-----------------------------------------------------------------------------------------
2.5 |    softsign    |  logcosh                         |    adam       |   0.8021      |
-----------------------------------------------------------------------------------------
2.6 |    tanh        |  logcosh                         |    adam       |   0.8414      |
-----------------------------------------------------------------------------------------
2.7 |    sigmoid     |  logcosh                         |    adam       |   0.9232      |
-----------------------------------------------------------------------------------------
2.8 |  hard_sigmoid  |  logcosh                         |    adam       |   0.9177      |
-----------------------------------------------------------------------------------------
2.9 |    linear      |  logcosh                         |    adam       |   0.8556      |
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
=========================================================================================

"""
