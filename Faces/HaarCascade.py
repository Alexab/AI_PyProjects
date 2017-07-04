"""
Эта программа детектирует лица на изображении с помощью каскадов Хаара.
Можно заменить классификатор "face-cascade", если нужно детектировать что-то другое.
Данный классификатор взят из "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
"""

import os
import cv2
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
path_to_image = 'data/billmurray.jpg'

img = cv2.imread(path_to_image)
gray = cv2.imread(path_to_image, 0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if os.path.isfile('/data/billmurray.jpg') == False:
    print("Файл с изображением не найден")

if img != None:
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Изображение не было загружено")