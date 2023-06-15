# Импортировать необходимые библиотеки
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

# Инициализировать видеозахват
cap = cv2.VideoCapture(0)

# Создать экземпляр детектора рук
detector = HandDetector(maxHands=1)

# Задать смещение и размер изображения
offset = 20
imgSize = 300

# Счетчик
counter = 0

# Папка для сохранения изображений
folder = 'Data/Y'

while True:
    # Считать кадр из видеозахвата
    ret, img = cap.read()

    # Найти руки на кадре
    hands, img = detector.findHands(img)

    if hands:
        # Взять первую найденную руку
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Создать белое изображение
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Обрезать изображение
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        # Рассчитать соотношение сторон
        aspectRatio = h / w

        if aspectRatio > 1:
            # Если соотношение сторон больше 1, изменить ширину
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Если соотношение сторон меньше или равно 1, изменить высоту
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Отображение обрезанного изображения
        cv2.imshow('img_crop', imgCrop)

        # Отображение белого изображения
        cv2.imshow('img_white', imgWhite)

    # Отображение исходного изображения
    cv2.imshow('img', img)

    # Обработка нажатия клавиши 's'
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Увеличение счетчика и сохранение изображения
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)