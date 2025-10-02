import cv2 as cv
import numpy as np 

cap = cv.VideoCapture(0)

while(True):
    ret,img=cap.read()
    if(ret == True):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        ubb=(35, 40, 40)
        uba=(95, 255, 255)

        mask = cv.inRange(hsv, ubb, uba)
        res = cv.bitwise_or(img, img, mask=mask)

        cv.imshow('img', img)
        cv.imshow('res', res)
        
        k =cv.waitKey(1) & 0xFF
        if k == 27 :
            break
    else:
        break

cap.release()
cv.destroyAllWindows()