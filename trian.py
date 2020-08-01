import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from tensorflow.keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd


trainData = np.load("./trainingData.npy")
trainLabel = np.load("./trainingLabel.npy")
testData = np.load("./testData.npy")
testLabel = np.load("./testLabel.npy")

print(trainData.shape)
print(trainLabel.shape)
print(testData.shape)
print(testLabel.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150,150,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu'))           
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='adam')
print(model.summary())

model.fit(trainData, trainLabel,batch_size = 32, epochs = 3, verbose=1,validation_data=(testData, testLabel))

model.save("brain-tumor-model.h5")

