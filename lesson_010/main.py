"""

@authro NoeVG
@about Working with images

"""

# Packages

# system
import sys

# OpenCV
import cv2

# Mathematil operations 
import numpy as np

def display():
    # Read the image
    image = cv2.imread('botones.jpg')
    imageGRAY = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imageHSV = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    
    


    while True:
        # Display image
        cv2.imshow('Frame',image)
        cv2.imshow('Gray',imageGRAY)
        cv2.imshow('HSV',imageHSV)

        # quit the program when 'esc' key is pressend
        k = cv2.waitKey( 0 ) & 0xFF
        if k == 27:
            break

        cv2.destroyAllWindows()

def segmentationHSV():
    # Get the images from file
    frame    = cv2.imread('botones.jpg')
    
    # Changed color spaces
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)

    #Created mask HSV low and high
    low_color = np.array([40,18,18],np.uint8)
    high_color = np.array([80,254,254],np.uint8)

    mask = cv2.inRange( frameHSV,low_color,high_color)

    # Upgrade
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)

    kernel = np.ones((4,4),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)

    maskVis = cv2.bitwise_and( frame, frame, mask = mask )

    while True:
        # Display RGB image
        cv2.imshow('Frame RGB',frame)
        cv2.imshow('Frame HSV',frameHSV)
        cv2.imshow('Mask',mask)
        cv2.imshow('Mask Segmentation',maskVis)

        # Quit program whenn 'esc' key in pressed
        k = cv2.waitKey(0) & 0XFF
        if k == 27:
            break

        cv2.destroyAllWindows()

def main():
    segmentationHSV()

if __name__ == "__main__":
    """
    Function to execute main
    """
    main()