import numpy as np
import cv2

cap = cv2.VideoCapture(0) #video capture token; 0 for the 1st webcam
cap.set(3,640) #this sets the Width; 3 corresponds to width
cap.set(4,480) #this sets the Height; 4 corresponds to the height 
 
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('BGRframe', frame)
    cv2.imshow('gray', gray) #this displays the result in a window
    
    #k = cv2.waitKey(30) & 0xff  #Bitwise anding
    #if k == 27: # press 'ESC' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()