"""
Код взят отсюда, только я не генерирую так много данных, я взяла изначально побольше картинок.
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# требуемый размер изображений
img_width, img_height = 150, 150
# папка с данными для обучения
train_data_dir        = 'data/train2/'
# папка с данными для тестирования
validation_data_dir   = 'data/test/'
# кол-во данных
nb_train_samples      = 4597+7643
nb_validation_samples = 25000

epochs                = 5
batch_size            = 16

nb_classes = 2

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(rescale = 1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen  = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
        'data/train',           # this is the target directory
        target_size=(150, 150), # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')    # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

#Сохраним модель
#model.save("CatsVsDogs_original.h5")

"""
!!!! я изменила переменные, хранящие кол-во данных, может быть другой результат!

                            РЕЗУЛЬТАТ (epochs = 5, почти 3 часа)
                            
         ======================================================================
         |                      |        ошибка         |     точность        |
         ======================================================================
         | на обучаюшей выборке |   loss: 0.4398        |   acc: 0.8128       |
         ----------------------------------------------------------------------
         | на тестовой выборке  |   val_loss: 0.3489    |   val_acc: 0.8506   |
         ======================================================================

"""
