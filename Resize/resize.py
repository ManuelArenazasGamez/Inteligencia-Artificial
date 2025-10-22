import numpy as np
import cv2 as cv
import math 

cap = cv.VideoCapture('f1.mp4')
i = 0  
while True:
    ret, frame = cap.read()
    frame2 = cv.resize(frame2, (28, 28), interpolation=cv.INTER_AREA)
        
    if(i%30==0):
        cv.imwrite('f1\\f1'+str(i)+'.jpg', frame2)

    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()