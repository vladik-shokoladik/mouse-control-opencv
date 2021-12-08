from cv2 import cv2
import mediapipe as mp
import numpy as np

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x * 
                flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                flippedRGB.shape[0])

        #TODO: results.multi_hand_landmarks[0].landmark[8].[x / y] преобразовать в координаты экрана
        cv2.circle(flippedRGB,(x_tip, y_tip), 10, (255, 0, 0), -1)
        # print(results.multi_hand_landmarks[0])
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()