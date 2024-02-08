# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:57:45 2023

@author: hatem
"""
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import  confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
data=pd.read_csv("Celiac disease.csv")

print(data.head())
print(data.isnull().sum())
print('=========================================================================')



#by looking at the dat we notice that the fiarst row was filled with nul vilues\
#the age coulmn have alot of null values so we chooses to get reed of them
data=data.drop(columns=(['age','Timestamp']))
data=data.iloc[1:]
value_counts = data['Medical diagnosis'].value_counts()
# Create a pie chart
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Adding title
plt.title('Pie Chart of Category Distribution')

# Display the chart
plt.show()
#lets check now
print(data.isnull().sum())
#now we are going to turn catogrical dat into numrical
#using to ways the OneHotEncoder and LabelEncoder
catorigical_features=data.drop(columns=['Medical diagnosis']).columns.tolist()
encoder = LabelEncoder()
data[catorigical_features] = data[catorigical_features].apply(encoder.fit_transform)
print(data.head())
print('=========================================================================')
#now split the data
x=data.drop(columns=['Medical diagnosis'])
y=data["Medical diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42) 

# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
 

# Use the SVM classifier to fit data:
classifier = SVC(kernel='liner')
# You can adjust this number as needed
num_features_to_select = 12 
rfe_selector = RFE(estimator=classifier, n_features_to_select=num_features_to_select) 
rfe_selector = rfe_selector.fit(X_train, y_train)
X_train_selected = rfe_selector.transform(X_train)
X_test_selected = rfe_selector.transform(X_test)
classifier.fit(X_train_selected, y_train)

# Predict y data with classifier: 
y_test_encoded=encoder.fit_transform(y_test)
y_predict = classifier.predict(X_test_selected)
y_probs = classifier.decision_function(X_test_selected)
# Calculate ROC curve and AUC:
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_probs)
roc_auc = roc_auc_score(y_test_encoded, y_probs)

# Print results: 
print('hold out')   
print(f"AUC: {roc_auc:.2f}")
print('=======================================================================')
print('confusion_matrix')    
print(confusion_matrix(y_test, y_predict))
print('======================================================================')
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")
print('======================================================================')



