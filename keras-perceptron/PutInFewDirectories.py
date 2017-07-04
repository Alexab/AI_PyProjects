import os
import shutil

"""
"""
input_dir           = '/home/sveuser/PycharmProjects/keras-perceptron/data/test/'
output_dir_template = '/home/sveuser/PycharmProjects/keras-perceptron/data/'
output_dir_cases    = ('cats/', 'dogs/')
filters             = ('cat', 'dog')

files = os.listdir(input_dir)

for fi in range(len(filters)):
    images = filter(lambda x: x.startswith(str(filters[fi])), files)
    target_dir = output_dir_template + str(output_dir_cases[fi])
    for i in images:
        shutil.move(input_dir + i, target_dir)

print("Перемещение успешно завершено")