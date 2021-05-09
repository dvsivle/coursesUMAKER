"""
@author Ing. Noé Vásquez Godínez
@about This practice about Canny with OpenCV
"""

# Import packages

# System
import sys

# OpenCV
import cv2

# Mathematical operations and structures
import numpy as np

def working():
    '''
    This function working with filters
    '''
    # Get information
    frame = cv2.imread("circulos.jpg")
    frameEtiquetado = frame.copy()
    # Changed the space colors
    frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Apply Gaussiano Filter
    gauss = cv2.GaussianBlur(frameGray, (5,5),0)
    
    # Get border with Canny
    canny = cv2.Canny(gauss,50,150)

    # Get the number borders
    (borders,_) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print("Objects: {} ".format(len(borders)))
                                      # B G R
    cv2.drawContours(frameEtiquetado,borders,-1,(0,255,0),9)

    while True:
        # Display RGB image
        cv2.imshow("Fase 1 Frame Original",frame)

        # Display Gray image
        cv2.imshow("Fase 2 Gray Frame",frameGray)

        # Display Frame with Filter
        cv2.imshow("Fase 3 Filter Frame",gauss)

        # Display Frame with canny
        cv2.imshow("Fase 4 Canny Frame",canny)

        # Display Frame with reds borders
        cv2.imshow("Fase Borders Red",frameEtiquetado)


        # Quit program when 'esc' key input
        k = cv2.waitKey(0) & OxFF
        if k == 27:
            break
        
        cv2.destroyAllWindows()

def main():
    '''
    This function init
    '''
    working()
    
if __name__ == "__main__":
    '''
    Function to execute main function
    '''
    main()