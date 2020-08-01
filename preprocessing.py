import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization, Activation
from keras import backend as K
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd


classes = os.listdir("./Training")
# need to save classes

enc = OneHotEncoder()
enc.fit([[0], [1], [2], [3]])


def names(number):
    if number == 0:
        return classes[0]
    elif number == 1:
        return classes[1]
    elif number == 2:
        return classes[2]
    elif number == 3:
        return classes[3]


trainData = []
trainLabel = []
dim = (150, 150)
trainPath = "./Training"
index = 0
txt = open("classes.txt", "w")  # write mode
for dir in os.listdir(trainPath):
    filePaths = []
    subDir = os.path.join(trainPath, dir)
    for file in os.listdir(subDir):
        imgFullPath = os.path.join(subDir, file)
        filePaths.append(imgFullPath)
        img = Image.open(imgFullPath).convert('LA')
        x = img.resize(dim)
        x = np.array(x)
        trainData.append(np.array(x))
        trainLabel.append(enc.transform([[index]]).toarray())
    print(names(index))
    txt.write(str(dir) + "\n")
    print(str(dir))
    index += 1
txt.close()

trainData = np.array(trainData)
trainLabel = np.array(trainLabel).reshape(2870, 4)

np.save("trainingData.npy", trainData)
np.save("trainingLabel.npy", trainLabel)
print(trainData.shape)
print(trainLabel.shape)


testData = []
testLabel = []
dim = (150, 150)
testPath = "./Testing"
index = 0
for dir in os.listdir(testPath):
    filePaths = []
    subDir = os.path.join(testPath, dir)
    for file in os.listdir(subDir):
        imgFullPath = os.path.join(subDir, file)
        filePaths.append(imgFullPath)
        img = Image.open(imgFullPath).convert('LA')
        x = img.resize(dim)
        x = np.array(x)
        testData.append(np.array(x))
        testLabel.append(enc.transform([[index]]).toarray())
    print(names(index))
    print(str(dir))
    index += 1
testData = np.array(testData)
testLabel = np.array(testLabel).reshape(394, 4)
print(testData.shape)
print(testLabel.shape)

np.save("testData.npy", trainData)
np.save("testLabel.npy", trainLabel)
