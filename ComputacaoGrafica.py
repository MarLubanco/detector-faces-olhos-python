import numpy as np
import cv2 as cv

detector_faces = cv.CascadeClassifier('/home/surfista/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
detector_olhos = cv.CascadeClassifier('/home/surfista/Downloads/opencv-master/data/haarcascades/haarcascade_eye.xml')
image = cv.imread('teste.jpeg')
image_cinza = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = detector_faces.detectMultiScale(image_cinza, 1.3, 5)

for (x,y,w,h) in faces:
    cv.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = image_cinza[y:x + h, x: x + w]
    roi_color = image[y:y+h, x:x+w]
    eyes = detector_olhos.detectMultiScale(roi_color)
    for(ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(255,0,0),2)

cv.imshow('imagem', image)
cv.waitKey(0)
cv.destroyAllWindows()


