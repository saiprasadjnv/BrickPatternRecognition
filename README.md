Training model:
To train the model with new dataset, import the "train" module and invoke the train method as follows:

from train import binaryClassification, trainData, testData, extract_features, SubClassModel, train
train(images, labels)
This method generates and serializes a trained model using the new dataset. 
The trained models are saved as:
classifier1.pt
classsifier2.pt


Test Model: 
To test the model, import the "test" module and invoke the "test" method as follows: 

from test import binaryClassification, trainData, testData, extract_features, SubClassModel, test
labels = test(images)
The method returns the predicted labels.
