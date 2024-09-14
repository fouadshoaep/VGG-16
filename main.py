import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
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
warnings.filterwarnings('ignore')
#########################################
path ='myData'
test_Ratio = 0.2
test_validation = 0.2
input_shape_VAL = 224
kernel_size_VAL = 3
print("Kernel Size",kernel_size_VAL)
print("Input Image Size",input_shape_VAL)
imageDimension=(input_shape_VAL,input_shape_VAL,3)
batchSizeVal=20
stepsPerEpochVal= 1067 # total items(6400) / batchsize (100)+1 = 64
epochsVal =20
#########################################
images =[]
classNo =[]
mylist = os.listdir(path)

noOfClasses = len(mylist)

print("total Number of Classes is :"+str(noOfClasses))
print("Import the Classes ..........")
for x in range(0,noOfClasses):
    myPiclist = os.listdir(path+"/"+str(x))
    for y in myPiclist:
        curImg = cv2.imread(path + "/" + str(x) + "/" + str(y))
        curImg = cv2.resize(curImg, (imageDimension[0], imageDimension[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")

print(" ")

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
# splitting th Data
x_train,x_test,y_train,y_test = train_test_split(images,classNo,test_size=test_Ratio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=test_validation)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

num_of_Samples = []
for x in range (0,noOfClasses):
    print(len(np.where(y_train==x)[0]))
    num_of_Samples.append(len(np.where(y_train==x)[0]))

print(num_of_Samples)



def preprossesing (img):
    img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255
    return img

print(x_train.shape)

print(x_train.shape)
img =x_train[31]
img = cv2.resize(img , (255,255))
print(y_train[31])
cv2.imshow("fff",img)
cv2.waitKey(0)

dataGen = tf.keras.preprocessing.image.ImageDataGenerator (width_shift_range=0.1,
                             height_shift_range=0.1,
                              zoom_range=0.2,
                             shear_range=0.1,
                            rotation_range=10)
dataGen.fit(x_train)
print(x_train.shape)


y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)


def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(input_shape_VAL , input_shape_VAL ,3), filters=64, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(kernel_size_VAL, kernel_size_VAL), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    #opt = Adam(learning_rate=0.001)
    SGDopt = keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.0,
    nesterov=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    loss_scale_factor=None,
    gradient_accumulation_steps=None,
    name="SGD",

)
    model.compile(optimizer=SGDopt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model


model1=VGG16()
print(model1.summary())

history = model1.fit(x_train ,y_train,batch_size=batchSizeVal,

                 epochs=epochsVal,
                 validation_data=(x_validation,y_validation),
                 shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss' , 'val_loss'] ,loc ='upper left' )
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'],loc ='upper left')
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
score = model1.evaluate(x_test,y_test,verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =  ', score[1])

model1.save("my_model.h5")
