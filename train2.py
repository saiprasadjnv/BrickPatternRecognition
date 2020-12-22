import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from skimage.feature import hog
import pickle
import glob
import numpy as np
from PIL import Image
import cv2
from matplotlib.pyplot import *
import random
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch import nn, optim
from sklearn.utils import shuffle
import torch


import torch.nn as nn
class SubClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution
        # 32 X 32
        self.conv1 = nn.Conv2d(1,3,11) ##190
        self.relu1 = nn.ReLU()     
        self.pool1 = nn.MaxPool2d(2) ## 95
        self.batchNorm1 = nn.BatchNorm2d(3) 
        self.dropOut1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(3,3,26) ## 70 
        self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
        self.batchNorm2 = nn.BatchNorm2d(3) 

        self.conv3 = nn.Conv2d(3,3,21) ## 50 
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) ## 25 
        self.batchNorm3 = nn.BatchNorm2d(3) 
        self.dropOut3 = nn.Dropout(p=0.1) 
        
        self.conv4 = nn.Conv2d(3,1,10) ## 16 
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2) ## 8
        self.batchNorm4 = nn.BatchNorm2d(1) 
        
#         self.conv5 = nn.Conv2d(3,3,21) 
#         self.relu5 = nn.ReLU()
# #         self.pool5 = nn.MaxPool2d(2)
#         self.batchNorm5 = nn.BatchNorm2d(3) 
        
        # self.conv6 = nn.Conv2d(3,1,10) 
        # self.relu6 = nn.ReLU()
        # self.pool6 = nn.MaxPool2d(2)
#         self.batchNorm6 = nn.BatchNorm2d(3) 
        
        self.fc1 = nn.Linear(1*8*8, 3, bias=True)
        
        self.outFn = nn.Softmax(dim=1)
        
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.pool1(x)
        x = self.batchNorm1(x)
        x = self.dropOut1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.pool2(x)
        x = self.batchNorm1(x)
       
        x = self.conv3(x)
        x = self.relu3(x) 
        x = self.pool3(x)
        x = self.batchNorm1(x)
        x = self.dropOut3(x)
       
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        # x = self.batchNorm2(x)
#         x = self.dropOut(x)

#         x = self.conv5(x)
#         x = self.relu5(x)
#         x = self.pool5(x)
#         x = self.batchNorm1(x)
        
        # x = self.conv6(x)
        # x = self.relu6(x)
        # x = self.pool6(x)
#         x = self.batchNorm1(x)
        
        x = x.view(-1,1*8*8)
        x = self.fc1(x)
        
        x = self.outFn(x)
        return x

# def getGrayScaleImages(images): 
#     grayImages =[]
#     for image in images: 
#         rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         grayImage = cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
#         grayImages.append([grayImage])
#     return np.array(grayImages)

def getGrayScaleImages(images): 
    grayImages =[]
    for image in images: 
        # rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # grayImage = cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        grayImage = color.rgb2gray(image)
        grayImages.append([grayImage])
    return np.array(grayImages)


def performEpoch(model, lossFn, optimizer, input_data, actual_output):
    model.zero_grad()
    prediction_output=model.forward(input_data)
    loss = lossFn(prediction_output,actual_output)
    loss.backward()
    optimizer.step()
    return prediction_output, loss


#### This function assumes that the classes are only 1,2,3
def train2(trainImages, trainLabels):
    if trainImages.shape[0] != trainLabels.shape[0]:
        return -1
    ## Prepare the dataset for training.
    ##### select images to train 
    imagesToTrain = []
    labelsToTrain = []
    for idx,x in enumerate(trainLabels):
        if x not in [0,4]:
            imagesToTrain.append(trainImages[idx])
            labelsToTrain.append(trainLabels[idx])
    trainImages = np.array(imagesToTrain)
    trainLabels = np.array(labelsToTrain) -1
    trainImages = getGrayScaleImages(trainImages)
    trainImages, trainLabels = shuffle(trainImages, trainLabels, random_state=0)
    trainImages = torch.tensor(trainImages, dtype=torch.float32)
    trainLabels = torch.tensor(trainLabels, dtype=torch.long)
    ## model 
    model = SubClassModel()
    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    l  = len(trainImages)
    for epoch in range(100):
        epoch_loss= 0 
        for i in range(50, l, 50):
            predicted_output, loss = performEpoch(model, lossFn, optimizer, trainImages[i-50:min(i,l)], trainLabels[i-50:min(i,l)]) 
            epoch_loss += loss 
    torch.save(model, "classifierTrain2.pt")
    print("trained classifier2")