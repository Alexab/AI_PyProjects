"""
В этой программе задача классификации кошки-собаки решается с помощью перцептрона
(в комментариях есть варианты для одного слоя, двух  и трёх).
Выходной вектор имеет вид:
[0., 1.] - для 100% вероятности кошки
[1., 0.] - для 100% вероятности собаки
Входная матрица для обучения и тестирования составляется из изображений в заданных директориях.

!!! Программа виснет на этапе составления этой матрицы, поскольку она, видимо, слишком большая !!!

Надо подумать, что можно сделать: преобразовать изображения в файлы csv, загружать частями и т.д.

"""

import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense

""" ПОДГОТОВИМ ИЗОБРАЖЕНИЯ """

train_input_dir = ('data/train2/cats/', 'data/train2/dogs/')
test_input_dir  = ('data/test/cats/', 'data/test/dogs/')
nb_classes = 2
width  = 150
height = 150

X_train = []
Y_train = []
X_test  = []
Y_test  = []

f = 0
for id in train_input_dir:
    print("Processing images from " + str(id))
    if f == 0:
        y = [0., 1.] #This is cat
    if f == 1:
        y = [1., 0.] #This is dog
    f += 1
    for ff in os.listdir(id):
        #read image into cv2 format
        image = cv2.imread(id + ff)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(width,height))
        image = np.ndarray.flatten(image)
        #use swapaxes to convert image to Keras' format
        #image_convert=np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)
        X_train.append(image)
        Y_train.append(y)
        #Show that it worked
        #print("Shape of image " + str(ff) + " is " + str(image.shape) + " now")
        #plot.imshow(image_convert[1,:,:])
        #plot.show()

f = 0
for id in test_input_dir:
    print("Processing images from " + str(id))
    if f == 0:
        y = [0., 1.] #This is cat
    if f == 1:
        y = [1., 0.] #This is dog
    f += 1
    for ff in os.listdir(id):
        #read image into cv2 format
        image = cv2.imread(id + ff)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(width,height))
        image = np.ndarray.flatten(image)
        #use swapaxes to convert image to Keras' format
        #image_convert=np.swapaxes(np.swapaxes(image, 1, 2), 0, 1)
        X_test.append(image)
        Y_test.append(y)
        #Show that it worked
        #print("Shape of image " + str(ff) + " is " + str(image.shape) + " now")
        #plot.imshow(image_convert[1,:,:])
        #plot.show()

Y_train = np.asarray(Y_train)
X_train = np.asarray(X_train)
Y_test = np.asarray(Y_test)
X_test = np.asarray(X_test)

print("X_train.shape", X_train.shape)
print("Y_train.shape", Y_train.shape)
print("X_test.shape", X_test.shape)
print("Y_test.shape", Y_test.shape)

""" СОСТАВИМ И СКОМПИЛИРУЕМ МОДЕЛЬ """

model = Sequential()
#для однослойного
#model.add(Dense(input_dim=width*height, activation="softmax", units=nb_classes))
#для двуслойного
model.add(Dense(input_dim=width*height, activation='relu', units=100))
model.add(Dense(units=nb_classes, input_dim=width*height, activation='softmax'))
#для трехслойного
#model.add(Dense(activation='relu', units=100, input_dim=width*height))
#model.add(Dense(activation='relu', units=200))
#model.add(Dense(activation='softmax', units=nb_classes))

#model.summary() #Print model info

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

""" ОБУЧИМ И ПРОВЕРИМ МОДЕЛЬ """

model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test,verbose=0)
print('Test accuracy:', score[1])

""" ВНИМАНИЕ: ПРИ ПЕРВОМ ЗАПУСКЕ И ПРИ ПОСЛЕДУЮЩИХ, ЕСЛИ НЕОБХОДИМО, РАСКОММЕНТИРОВАТЬ СОХРАНЕНИЕ МОДЕЛИ """
#Сохраним модель
#model.save("CatVsDogs_mymodel.h5")