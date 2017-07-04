from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Подготовим изображения

img_width, img_height = 150, 150
input_dir0            = 'data/train/cats/'
output_dir0           = 'data/train_generated/cats/'
input_dir1            = 'data/train/dogs/'
output_dir1           = 'data/train_generated/dogs/'
input_format          = 'jpeg'
output_format         = 'jpeg'
batch_size            = 16
nb_train_samples      = 25000
nb_validation_samples = 25000
epochs                = 5
nb_classes            = 2

image_datagen = ImageDataGenerator(rescale = 1. / 255)
image_list    = image_datagen.flow_from_directory(
        input_dir0,
        target_size = (img_width, img_height),
        batch_size  = batch_size,
        classes     = ('cats', 'dogs'),
        save_to_dir = output_dir0)

#Загружаем список файлов из указанной папки и конвертируем в массив
#format_files0 = filter(lambda x: x.endswith(str(input_format)), os.listdir(input_dir0))
format_files0 = os.listdir(input_dir0)

for ff in format_files0:
    img = load_img(input_dir0 + ff)  # this is a PIL image
    x = img_to_array(img)            # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)    # this is a Numpy array with shape (1, 3, 150, 150)
    print(x.shape)
    for i in image_datagen.flow(x, batch_size=1, save_to_dir=output_dir0, save_format=output_format):
            print(x.shape)
            break;

"""
#Сохраним в указанную папку
image_datagen = ImageDataGenerator(rescale = 1. / 255)
image_list = image_datagen.flow_from_directory(
        input_dir0,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes = ('cats', 'dogs'))


print(len(image_list))

#X_train0 = [np.array(x) for x in image_list]
#print(X_train0.shape)


model = Sequential()
#для однослойного
#model.add(Dense(input_dim=784, activation="softmax", units=nb_classes))
#для двуслойного
#model.add(Dense(input_dim=784, activation='relu', units=100))
#model.add(Dense(units=nb_classes, input_dim=784, activation='softmax'))
#для трехслойного
model.add(Dense(activation = 'relu',    units = 100, input_shape=input_shape))
model.add(Dense(activation = 'relu',    units = 200))
model.add(Dense(activation = 'softmax', units = nb_classes))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# this is the augmentation configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(rescale = 1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen  = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir='data/train_generated')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir='data/test_generated')

model.fit_generator(
        train_generator,
        steps_per_epoch=25000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=25000 // batch_size)
#model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=2, validation_data=(X_test, Y_test))

#score = model.evaluate(X_test, Y_test,verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

#Сохраним модель
model.save("catsVSdogs_model.h5")
"""
