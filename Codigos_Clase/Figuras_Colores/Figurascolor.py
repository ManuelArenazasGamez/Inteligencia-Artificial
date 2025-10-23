import cv2 as cv
import numpy as np

img = cv.imread('figura2.png', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
ubb=(35, 100, 100)
uba=(85, 255, 255)
#ubb2=(170, 100, 100)
#uba2=(180, 255, 255)

mask1 = cv.inRange(hsv, ubb, uba)
#mask2 = cv.inRange(hsv, ubb, uba), #ubb2, uba2)
mask = mask1 #+ mask2

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    print(f"Se encontraron {len(contours)} contornos.")
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)

        if area > 100:  
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv.circle(img, (cX, cY), 5, (0, 255, 0), -1) 
                cv.putText(img, f"({cX}, {cY})", (cX + 10, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Mancha {i+1} - Coordenadas del centro: ({cX}, {cY})")
else:
    print("No se encontraron")

res = cv.bitwise_and(img, img, mask=mask)
cv.imshow('mask', mask)
cv.imshow('res', res)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()