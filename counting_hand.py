import time

import cv2
import numpy as np
import os
import hand as htm

# 0 lay camera hien tai
cap = cv2.VideoCapture(0)
# khai bao diem bat dau thoi gian = 0
ptime = 0
# khai bao font chu
font = cv2.FONT_HERSHEY_PLAIN
folderPath = "Fingers"
lst = os.listdir(folderPath)
lst2 = []
for i in lst:
    # f de doi no thanh string {folderPath} / {i} de lay duong dan -> xu li cac anh lay
    img = cv2.imread(f"{folderPath}/{i}")
    lst2.append(img)
    #print(f"{folderPath}/{i}")
print(lst2[0].shape)

# khai bao bien de no phat hien
detector = htm.handDetector(detectionCon= int(0.5))

fingerID = [4,8,12,16,20]
while True:
    ret, frame = cap.read()
    # w rong, h cao, c chanel
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    #print(lmList)


    #viet cho ngon dai
    # chi khi phat hien ra khi no phat hien ra ban tay
    if len(lmList)!=0:
        # ve hinh chu nhat hien so ngon tay
        cv2.rectangle(frame, (0, 200), (130, 400), (255, 255, 255), -1)

        # [2]: su dung theo chieu cao de so sanh
        # [8], [6]: diem cua ngon tro trong so do mediapipe

        fingers = []
        # viet cho ngon cai: (diem 4 nam ben trai hay phai)
        # [1]: chieu rong
        if lmList[fingerID[0]][1] < lmList[fingerID[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        # viet cho ngon dai:
        for id in range(1,5):

            if lmList[fingerID[id]][2] < lmList[fingerID[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        soNgonTay = int(fingers.count(1))
        cv2.putText(frame, str(soNgonTay), (50, 310), font, 3, (0, 0, 0), 3)
        w, h, c = lst2[soNgonTay].shape
        frame[0:w, 0:h] = lst2[soNgonTay]






    #fps so khung hinh tren 1 giay
    # time.time() diem bat dau thoi gian
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    #show fps len man hinh
    cv2.putText(frame, f"FPS: {int(fps)}", (150,50), font, 2, (0,0,0), 3)

    cv2.imshow("cam", frame)
    if cv2.waitKey(1) == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()