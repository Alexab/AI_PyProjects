"""
Эта программа предсказывает результат для одного изображения, используя модель "CatsVsDogs_original.h5",
сохранённую в результаты работы программы "example2_1CatsVsDogs_original"
"""

import cv2
import os
import numpy as np
from keras.models import load_model

#Путь к папке, в которой лежат изображения для предсказания.
#ТАМ НЕ ДОЛЖНО БЫТЬ ФАЙЛОВ, НЕ ЯВЛЯЩИХС


# Я ИЗОБРАЖЕНИЯМИ!
image_dir = "data/predict/"

#Ширина-высота зависят от параметров модели, не надо их менять!
width = 150
height = 150

""" Загружаем сохранённую ранее модель """
model = load_model("CatsVsDogs_original.h5")

""" Пробуем предсказать """
for f in os.listdir(image_dir):
    """ Подготовим изображение """
    image = cv2.imread(image_dir + f)
    #уменьшим изображение.если оно слишком большое
    if image.shape[0] > 400:
        image = cv2.resize(image, (image.shape[1]*400//image.shape[0], 400))
    if image.shape[1] > 600:
        image = cv2.resize(image, (600, image.shape[0]*600//image.shape[1]))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    convert_image = cv2.resize(image, (width, height))
    convert_image = convert_image.reshape((1,) + convert_image.shape)

    #our_images = our_images.astype("float32")
    result = np.ndarray.flatten(model.predict(convert_image))

    if result[0] < 0.2:
        answer = "Это кот!"
    else:
        answer = "Это собака!"

    #cv2.putText(image, answer, (50, 60), cv2.FONT_ITALIC, 0.25, (255, 255, 255), 2)
    cv2.imshow('Кто же это - кот или собака?', image)

    print("Result = " + str(result) + "; \t" + answer + "\r"),

    pressed_key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if pressed_key != 32:   # 32 is a spacebar
        break

cv2.destroyAllWindows()
