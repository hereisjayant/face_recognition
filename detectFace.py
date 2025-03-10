import numpy as np
import cv2

#cascade used for the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
cap.set(3,1280) # set Width
cap.set(4,720) # set Height

while 1:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    faces = face_cascade.detectMultiScale(
        gray,     #source
        scaleFactor=1.3, #used to create the scale pyramid
        minNeighbors=5,     #A higher number gives lower false positives.
        minSize=(30, 30)   #minimum rectangle size ti be considered a face
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #(image, start_point, end_point, color, thickness)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        


    cv2.imshow('Output',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()


