import numpy as np
from keras.models import load_model
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D ,MaxPooling2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
import keras
import tensorflow as tf
import warnings
import pickle
########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.1# MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
imagesize=224
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL
#pickle_in = open("A\model_trained.keras","rb")
#model = pickle.load(pickle_in)
model = load_model("my_model.h5")
#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    #success, imgOriginal = cap.read()
    imgOriginal=cv2.imread("D:\\master\\myData\\8\\Eight_full (19).jpg")
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(imagesize,imagesize))
    #img = preProcessing(img)
    #cv2.imshow("Processsed Image",img)
    img = img.reshape(1,imagesize,imagesize,3)
    #### PREDICT
    classIndex = np.argmax(model.predict(img), axis=-1)
    #print(classIndex)
    #print(classIndex)
    predictions = model.predict(img)
    print(predictions)
    probVal= np.amax(predictions)
    print(probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
    cv2.resize(imgOriginal,(1500,1500))
    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break