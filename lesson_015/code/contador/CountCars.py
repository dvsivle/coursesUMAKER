# System
import sys

#OpenCV
import cv2

#Mathematical and structure
import numpy as np

# Plot charts
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

# Tensorflow model load
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.models import model_from_json


# Load setup json model

json_file = open('model.json','r')
loaded_model_json = json_file.read()

json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

print("Model load from PC")

# compile
model.compile(optimizer=RMSprop(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics = ['acc']
            )

# Plot area
#areas = []

# Flag to count cars

actual = False
anterior = False

actual2 = False
anterior2 = False


#Setup text
text = "Cars: "
position = (100,100)
position2 = (600,100)

font = cv2.FONT_HERSHEY_TRIPLEX
fontSize = 3
color  = (0,255,0)
color2 = (255,0,0)

thickness = 10

totalcars1 = 0
totalcars2 = 0

# Get information

cap = cv2.VideoCapture("NR_Curva.mp4")

while( cap.isOpened() ):
    # Capture information
    ret, fframe = cap.read()
    
    if ret == True:
        frame = fframe.copy()
        
        #Frame changed Gray
        gray = cv2.cvtColor(fframe,cv2.COLOR_BGR2GRAY)

        #Get size of the original frame
        rows = fframe.shape[0]
        cols = fframe.shape[1]

        # Get points to ROI
        x1 = (cols//2)-220
        x2 = (cols//2)-100
        y1 = (rows//2)-200
        y2 = rows//2

        ROI2x1 = (cols//2)-40
        ROI2x2 = (cols//2)+100
        ROI2y1 = (rows//2)-200
        ROI2y2 = rows//2


        cv2.line(frame,(0,y2),(cols,y2),(0,0,255),10)

        foto = fframe[(rows//2)-30:(rows//2)+200,(cols//2)-300:(cols//2)-10]

        foto2 = fframe[(rows//2)-30:(rows//2)+200,(cols//2):(cols//2)+250]

        #Create frame auxiliary image to remove backgroud
        imAux = np.zeros(shape=(gray.shape[:2]),dtype=np.uint8)
        imAux2 = np.zeros(shape=(gray.shape[:2]),dtype=np.uint8)

        #Set the ROI to apply bitwise AND and get only car
        cv2.rectangle(imAux,(x1,y1),(x2,y2),(255,255,255),-1)
        cv2.rectangle(imAux2,(ROI2x1,ROI2y1),(ROI2x2,ROI2y2),(255,255,255),-1)

        #Apply the operation to get Mask
        image_area = cv2.bitwise_and(gray,gray,mask=imAux)
        image_area2 = cv2.bitwise_and(gray,gray,mask=imAux2)

        #Apply filter to procesing
        image_area = cv2.GaussianBlur(image_area,(21,21),0)
        image_area2 = cv2.GaussianBlur(image_area2,(21,21),0)

        #Binarize image
        ret,image_area = cv2.threshold(image_area,127,255,cv2.THRESH_BINARY)
        ret,image_area2 = cv2.threshold(image_area2,127,255,cv2.THRESH_BINARY)

        #Apply Morphological Operators
        kernel = np.ones((40,40),np.uint8)
        image_area = cv2.dilate(image_area,kernel,iterations=1)
        image_area2 = cv2.dilate(image_area2,kernel,iterations=1)

        # Point to validate object detection
        punto = 0
        punto2 = 0

        #Find contours
        cnts = cv2.findContours(image_area,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts2 = cv2.findContours(image_area2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

        #Check possible objects
        for cnt in cnts:            
            #Get the area object
            areaContornos = cv2.contourArea(cnt)
            
            if(areaContornos > 10000):
                #Draw all contours
                x, y , w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.circle(frame,(x,y+h),10,(0,255,0),-1)
                punto = y + h

            #print("Area de este contorno: ",areaContornos)
            #areas.append(areaContornos)
        
        for cnt in cnts2:
            #Get the area object 
            areaContornos = cv2.contourArea(cnt)

            #if(areaContornos > 10000):
            if(areaContornos > 10000):
            
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
                cv2.circle(frame,(x,y+h),10,(255,0,0),-1)
                punto2= y + h 

        # If the point crossed the line, save state
        if( punto > y2+1 ):
            #Set status True
            actual = True
        else:
            actual = False
        
        # Get the possible car
        if(anterior == True and actual == False):
            #Set Frame to TensorFlow and validate if its car
            print("Hey! check with Tensorflow !!!")
            print(" Is this car ?")
            cv2.line(frame,(400,y2),(cols-700,y2),(0,255,0),5)
            #Format
            inputFrame = cv2.resize(foto,dsize=(150,150),interpolation = cv2.INTER_CUBIC)
            #Numpy array
            np_image_data = np.asarray(inputFrame)
            np_final = np.expand_dims(np_image_data,axis=0)
            predictions = model.predict(np_final)
            print(predictions)
            if(predictions <=0):
                print("Yes")
                totalcars1 +=1
            else:
                print("Not")
        
        anterior = actual

        # If the point crossed the line, save state
        if(punto2 > ROI2y2+1):
            #Set status True
            actual2 = True
        else:
            actual2 = False

        # Get the possible car
        if(anterior2 == True and actual2 == False):
            #Set Frame to Tensorflow and validate of its car
            print("Hey ! check with Tensroflow !!!")
            print("Is this car ?")
            cv2.line(frame,(600,ROI2y2),(cols-500,ROI2y2),(255,0,0),5) 
            #Format
            inputFrame = cv2.resize(foto2,dsize=(150,150),interpolation=cv2.INTER_CUBIC)
            #Numpy
            np_image_data = np.asarray(inputFrame)
            #convertion
            np_final = np.expand_dims(np_image_data,axis=0)
            predictions = model.predict(np_final)
            print(predictions)
            if(predictions[0][0]<=0 ):
                print("Yes")
                totalcars2 += 1
            else:
                print("Not")

        anterior2 = actual2

        cv2.putText(frame,text+str(totalcars1),position,font,fontSize,color,thickness)
        cv2.putText(frame,text+str(totalcars2),position2,font,fontSize,color2,thickness)


        # Frame shown
        cv2.imshow("Frame",frame)
        cv2.imshow("Carril 1 ",foto)
        cv2.imshow("Carril 2 ",foto2)
        
        #cv2.imshow("Frame Gray",gray)
        #cv2.imshow("Area ",image_area)
        


        k = cv2.waitKey(10) & 0xFF
        
        if(k == 27):
            break
        
    else:
        break

cv2.destroyAllWindows()
print("Total cars: ",totalcars1 + totalcars2)

# Plot area
#plt.plot(areas)
#plt.title("Area de objetos")
#plt.xlabel("Time (s)")
#plt.ylabel("Area values")
#plt.show()