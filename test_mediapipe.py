import cv2 as cv
import mediapipe as mp
import math
import numpy as np

cap = cv.VideoCapture(0)
model_path = "model/gesture_recognizer.task"
hands = mp.solutions.hands.Hands(static_image_mode=False,
                                 max_num_hands=2,
                                 min_tracking_confidence=0.5,
                                 min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
while True:
    ret, frame = cap.read()
    result = hands.process(frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS
            )

    cv.imshow("frame", frame)

    if cv.waitKey(1) == ord("q"):
        break

hands.close()
cap.release()
cv.destroyAllWindows()
