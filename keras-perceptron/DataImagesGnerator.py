"""
Эта программа берет изображения из input_dir, трансформирует их, затем сохраняет в папку output_dir.
Папки должны существовать.
Если нужно, можно сделать возможность работать с несколькими папками, а также создавать их в случае необходимости.
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

input_dir     = 'data/train_generated2/dogs/'
input_format  = 'jpeg'
output_dir    = 'data/train2/dogs/'
output_prefix = 'dog'
output_format = 'jpeg'
#при 0 генератор лишь сохранит по одному изменённому варианту каждого изображения,
#при 1 каждому исходному изображению будет соответствовать 2 изменённых варианта и так далее
number_of_variations_for_each_image = 5

#Сам генератор
datagen = ImageDataGenerator(
    rotation_range     = 40,        #диапазон (в градусах) для рандомного поворота
    width_shift_range  = 0.2,       #диапазон доли от ширины изображения для рандомного смещения по горизонтали
    height_shift_range = 0.2,       #диапазон доли от высоты изображения для рандомного смещения по вертикали
    rescale            = 1./255,    #на это умножаются все данные (элементы массива) перед всеми другими трансформациями
    shear_range        = 0.2,       #угол сдвига против часовой стрелки (в радианах)
    zoom_range         = 0.2,       #максимально допустимое отклонение в - и + для увеличения
    horizontal_flip    = True,      #разрешение отражать изображение по горизонтали
    fill_mode          = 'nearest') #как заполнять пустоты вокруг изображения  {"constant", "nearest", "reflect" or "wrap"}

#Получаем список файлов во входной папке
files = os.listdir(input_dir)
#Оставляем в этом списке только файлы заданного формата
#print(files)
#files = list(filter(lambda x: x.endswith(str(input_format)), files))
print("Найдено ", str(len(list(files))) + " файлов: ")
print(files)

"""Для каждого подходящего файла-изображения:
    - загружаем его как img
    - преобразуем в массив
    - меняем форму этого массива
    - применяем к нему генератор и сохраняем в выходную папку полученные изображения
"""
for ff in files:
    img = load_img(input_dir + ff)  # this is a PIL image
    x = img_to_array(img)           # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)   # this is a Numpy array with shape (1, 3, 150, 150)

    i = 0
    #.flow() генерирует рандомно измененное изображение и сохраняет его в указанную папку
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_dir, save_prefix=output_prefix, save_format=output_format):
        i += 1
        if i > number_of_variations_for_each_image:
            break  # otherwise the generator would loop indefinitely

print("Генерация изображений завершена")