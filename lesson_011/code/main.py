"""
@author Ing. Noé Vásquez Godínez
@about This example practice with 
filters pre-procesing images
"""

# Packages

# System
import sys

# OpenCV
import cv2

# Mathematical operations
import numpy as np

def working():
    '''
    This function working with filters
    '''
    # Get the images from file
    frame = cv2.imread('placas.jpg')

    # Geometric Transformations
    cols = frame.shape[1]
    rows = frame.shape[0]

    # Traslation
    '''
        | 1 0 tx|
    M = 
        | 0 1 ty|
    '''
    #M = np.float32( [[1,0,100],[0,1,150]] )
    #frameResult = cv2.warpAffine(frame,M,(cols,rows))

    # Rotation
    # M = cv2.getRotationMatrix2D( (cols//2,rows//2),45,1 )
    #frameResult = cv2.warpAffine(frame,M,(cols,rows))

    # Scale / Resize
    #frameResult = cv2.resize(frame,(800,800),interpolation=cv2.INTER_CUBIC)

    frameResult = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameResultGray = frameResult
    ret,frameResult = cv2.threshold(frameResult,180,255,cv2.THRESH_BINARY_INV)
    #ret,frameResult = cv2.threshold(frameResult,85,200,cv2.THRESH_BINARY)
    #ret,frameResult = cv2.threshold(frameResult,127,255,cv2.THRESH_TRUNC)
    #ret,frameResult = cv2.threshold(frameResult,100,200,cv2.THRESH_TOZERO)
    #ret,frameResult = cv2.threshold(frameResult,127,255,cv2.THRESH_TOZERO_INV)
    #frameBi = frameResult

    #kernel = np.ones((10,10),np.uint8)
    #frameResult = cv2.dilate(frameResult,kernel,iterations = 1)

    kernel = np.ones((12,12),np.uint8)
    frameResultErosion = cv2.erode(frameResult,kernel,iterations = 1)

    kernel = np.ones((10,10),np.uint8)
    frameResultDila = cv2.dilate(frameResultErosion,kernel,iterations = 1)

    ret,frameResultDila = cv2.threshold(frameResultDila,180,255,cv2.THRESH_BINARY_INV)

    #kernel = np.ones((15,15),np.uint8)
    #frameResultErosion = cv2.dilate(frameResultDila,kernel,iterations = 1)


    while True:
        # Display RGB image
        cv2.imshow('Image Original',frame)
        cv2.imshow('Image Gray',frameResultGray)
        #cv2.imshow('Image Bi',frameResult)
        #cv2.imshow('Image Result',frameResultErosion)
        cv2.imshow('Image Result di',frameResultDila)
        
        

        #cv2.imshow('Image Result Dilatación',frameResult)
        #cv2.imshow('Image Result Erosión',frameResultErosion)
        #Quit progra when 'esc' key in pressed
        k = cv2.waitKey(0) & 0XFF
        if k == 27:
            break
        cv2.destroyAllWindows()

def main():
    '''
    This start 
    '''
    working()

if __name__ == "__main__":
    """
    Function to execute main function
    """
    main()