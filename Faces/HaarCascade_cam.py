'''
Эта программа работает с веб-камерой - детектирует лица с помощью каскадов Хаара.
Можно заменить классификатор "face-cascade", если нужно детектировать что-то другое.
Окно с выводом веб-камеры закроется ТОЛЬКО ПО НАЖАТИЮ <ESC>! (я пока не знаю, как сделать это по крестику).
Параметр mirror определяет, отражать ли картинку по горизонтали.

cv2.waitKey(1) == 27    esc
cv2.waitKey(1) == 13    enter
cv2.waitKey(1) == 32    spacebar
'''

import cv2
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

# отразить ли изображение по горизонтали
mirror = True

while True:
    ret_val, img = cam.read()
    if mirror:
        img = cv2.flip(img, 1)

    #Здесь начинается обработка кадра
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #Здесь заканчивается обработка кадра

    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
