"""

@autor UMAKER | Ing. Noé Vásquez Godínez
@about Detección de esquinas

"""

# Import packages 

# System 
import sys

#OpenCV (2v)
import cv2

#Structures and operations
import numpy as np

def working():
    '''
    Esta funcion se ecnarga de procesar e inferir en nuestra imagen
    '''
    #Get information
    frame = cv2.imread('pos1.jpg') # BGR != RGB
    frameOrignal = frame.copy() 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    grayOriginal = gray.copy() 
    
    #Changed the structure to frame in gray
    gray = np.float32(gray)

    frameInfo = cv2.cornerHarris(gray,3,3,0.04)
    
    height, width =  frameInfo.shape

            # B G R
    color = (0,0,255) 

    for y in range(0,height):
        for x in range(0,width):
            if frameInfo.item(y,x) > 0.01 * frameInfo.max():
                cv2.circle(frame,(x,y),10,color,cv2.FILLED,cv2.LINE_AA)

    while True:
        # Display RGB image
        cv2.imshow("Original frame",frameOrignal)
        
        # Display Gray image
        cv2.imshow("Gray frame",grayOriginal)
        
        # Display harris
        cv2.imshow("Harris Info",frameInfo)

        # Display Frame
        cv2.imshow("Image with corner",frame)

        # Quit program when 'esc' key input
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break

        cv2.destroyAllWindows()

def main():
    '''
    This function init the all working
    '''
    working()

if __name__ == "__main__":
    '''
    Function to execute main function
    '''
    main()