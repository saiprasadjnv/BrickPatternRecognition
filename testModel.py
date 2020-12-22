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
import test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas import DataFrame

from test import binaryClassification, trainData, testData, extract_features, SubClassModel, test

images = np.load("finalTestingImages.npy")
true_labels = np.load("finalTestingLabels.npy")
predictedLabels = test(images)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Confusion Matrix: ")
print(DataFrame(confusion_matrix(true_labels, predictedLabels))) 
print("Accuracy: ", accuracy_score(true_labels,predictedLabels))
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")