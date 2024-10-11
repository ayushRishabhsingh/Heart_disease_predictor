#standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
import pickle

#1. get data ready
import pandas as pd
heart_disease=pd.read_csv("heartdisease.csv")
heart_disease

# Create X (feature matrix or data )
X=heart_disease.drop("target", axis=1)

# Create Y (labels or result or req output that is to be predicted)
Y=heart_disease["target"]

#2 choose the right model and hyperparameters for your data

#the hyperparameters are the settings to tune the model acco to data needs
clf=RandomForestClassifier()#its a model

# training of data or fit the model to training data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.2)#this means out of 10 data values only 8 will be used in training and 2 will be  used in testing

clf.fit(X_train,Y_train)

Y_preds=clf.predict(X_test)

print("\nClassification_report\n")
print(classification_report(Y_test,Y_preds))
print("\nConfusion_matrix:\n")
print(confusion_matrix(Y_test,Y_preds))
print("\nAccuracy:\n")
print(accuracy_score(Y_test,Y_preds)*100)

#6 save your model and load it
pickle.dump(clf, open("random_forest_model_1.pkl","wb"))








