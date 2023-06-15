import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, min_tracking_confidence=0,
                 min_detection_confidence=0.5):
        """
        Конструктор класса handDetector.

        Args:
            static_image_mode (bool): Режим обработки статического изображения.
            max_num_hands (int): Максимальное количество рук для обнаружения.
            min_tracking_confidence (float): Минимальное доверие к слежению.
            min_detection_confidence (float): Минимальное доверие к обнаружению.
        """
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detectionCon = min_tracking_confidence
        self.trackCon = min_detection_confidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Обнаруживает руки на изображении.

        Args:
            img (numpy.ndarray): Входное изображение.
            draw (bool): Флаг для рисования обнаруженных рук на изображении.

        Returns:
            numpy.ndarray: Изображение с нарисованными обнаруженными руками.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Определяет позицию ключевых точек руки на изображении.

        Args:
            img (numpy.ndarray): Входное изображение.
            handNo (int): Индекс руки для определения позиции.
            draw (bool): Флаг для рисования ключевых точек на изображении.

        Returns:
            list: Список с координатами ключевых точек руки.
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    """
    Основная функция программы.
    """
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()