"""
@author Ing. Noe Vásquez
"""

# Import packages

# System
import sys

# OpenCV
import cv2

#Structures 
import numpy as np

def working():
    '''
    This funtion detect faces
    '''

    face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('files/haarcascade_eye.xml')
    
    frame = cv2.imread('woman1.png')
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale( gray,1.1,5 )
    
    for (x,y,w,h) in faces:
        
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)        
        
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),10)

    while True:
        #Display RGB image
        cv2.imshow("Woman",frame)

        #Display Gray image
        cv2.imshow("Gray Woman",gray)

        #Quit program when 'esc' key input
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        cv2.destroAllWindows()

def main():
    '''
    This function init
    '''
    working()

if __name__ == "__main__":
    '''
    Function init
    '''
    main()
