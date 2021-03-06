from cv2 import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors

# get main moonitor resolution
MonitorH, MonitorW = 0, 0
for m in get_monitors():
    MonitorH, MonitorW = m.height, m.width
    # print(MonitorH, MonitorW)


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

        cursorX = results.multi_hand_landmarks[0].landmark[8].x * MonitorH
        cursorY = results.multi_hand_landmarks[0].landmark[8].x * MonitorW
        # print(cursorX, cursorY)

        cv2.circle(flippedRGB,(x_tip, y_tip), 10, (255, 0, 0), -1)
        


    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()