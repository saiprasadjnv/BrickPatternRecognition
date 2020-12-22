import glob
import math
import pickle
import random
from skimage import color
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.pyplot import *
from PIL import Image
from skimage.feature import hog
from sklearn import datasets
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, plot_confusion_matrix)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(1152, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def extract_features(images) :
    X = np.empty((0,1152),dtype='i2')
    for image in images:
        fd1, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)   
        for idx,z in enumerate(fd1):
          if (z == 0.0):
            fd1[idx] = 0.02 
        X = np.append(X,np.reshape(fd1,(1,-1)),axis=0)

    return X

def convert_to_binary_labels(labels):
    bin_y = []
    for i in range(0, len(labels)):
        if labels[i] != 0:
            bin_y.append(1)
        else:
            bin_y.append(0)
    return bin_y

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


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
        # print("Before conv: ", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        # print("Conv1: ", x.shape)
        x = self.pool1(x)
        # print("pool: ", x.shape)
        x = self.batchNorm1(x)
        x = self.dropOut1(x)

        # print("bNorm1: ", x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.pool2(x)
        x = self.batchNorm1(x)
        # print("Conv2: ", x.shape)
       
        x = self.conv3(x)
        x = self.relu3(x) 
        x = self.pool3(x)
        x = self.batchNorm1(x)
        x = self.dropOut3(x)
        # print("Conv3: ", x.shape)
       
        x = self.conv4(x)
        x = self.relu4(x)
        # print("Conv4: ", x.shape) 
        x = self.pool4(x)
        
        x = x.view(-1,1*8*8)
        x = self.fc1(x)
        
        x = self.outFn(x)
        return x


def getGrayScaleImages(images): 
    grayImages =[]
    for image in images: 
        # rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # grayImage = cv2.cvtColor(rgbImage,cv2.COLOR_RGB2GRAY)
        grayImage = color.rgb2gray(image)
        grayImages.append([grayImage])
    return np.array(grayImages)

images = [] 
labels = [] 


def classfier2(): 
    global images, labels
    model = torch.load("classifier2.pt")
    toDo = []
    for idx,y in enumerate(labels):
        if y==4:
            toDo.append(idx)
    testImages = [images[i] for i in toDo] 
    testImages = getGrayScaleImages(testImages)
    testImages = torch.tensor(testImages, dtype=torch.float32) 
    pred_probs = model(testImages).detach().numpy()
    for idx,x in enumerate(pred_probs):
        t = np.argmax(x) 
        if t==0 and max(x)>=0.91:
            labels[toDo[idx]] = 1
        elif t==1 and max(x)>=0.92:
            labels[toDo[idx]] = 2 
        elif max(x)> 0.96: 
            labels[toDo[idx]] = 3

def classifier1():
    global EPOCHS, BATCH_SIZE, LEARNING_RATE, images, labels
    model = torch.load("classifier1.pt").cpu()
    trainImages = images
    X_test = extract_features(trainImages)
    test_data = testData(torch.FloatTensor(X_test))
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            # X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    count = 0
    if np.isnan(y_pred_list).any():
        for idx, q in enumerate(y_pred_list):
            if (math.isnan(q)):
                count = count + 1
                y_pred_list[idx] = 1
    for idx, label in enumerate(y_pred_list):
        if (label == 1):
            y_pred_list[idx] = 4
    labels = y_pred_list
    
def test(testimages): 
    global images, labels
    images = testimages
    classifier1()
    classfier2()
    return labels
