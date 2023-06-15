# Импортировать необходимые библиотеки
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Инициализировать видеозахват
cap = cv2.VideoCapture(0)

# Создать экземпляр детектора рук
detector = HandDetector(maxHands=1)

# Создать экземпляр классификатора
classifier = Classifier("Model3/keras_model.h5", "Model3/labels.txt")

# Задать смещение и размер изображения
offset = 20
imgSize = 305

# Счетчик
counter = 0

# Папка для сохранения данных
folder = 'Data/C'

# Список меток классов
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

while True:
    # Считать кадр из видеозахвата
    ret, img = cap.read()

    # Создать копию кадра для вывода результата
    imgOutput = img.copy()

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
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            # Если соотношение сторон меньше или равно 1, изменить высоту
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Вывести метку класса на изображении
        cv2.putText(imgOutput, labels[index], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 0, 0), 2)

    # Вывести кадр с результатами
    cv2.imshow('img', imgOutput)

    # Если нажата клавиша 'q', выйти из цикла
    if cv2.waitKey(1) == ord("q"):
        break

# Освободить видеозахват и закрыть окна
cap.release()
cv2.destroyAllWindows()