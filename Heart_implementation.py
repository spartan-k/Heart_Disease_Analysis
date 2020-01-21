# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:06:44 2019

@author: Kartikeya
"""

#Importing all libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy

heart_data = pd.read_csv(r"C:\Users\Kartikeya\OneDrive - McGill University\Warut\heart.csv")

#Defining predictors and target
y = heart_data['target']
heart_data.drop(labels="target", axis=1, inplace=True)
X = heart_data

#Standardizing
from sklearn.preprocessing import StandardScaler
standardizer= StandardScaler()
X_std = standardizer.fit_transform(X)

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std ,y, test_size=0.3, random_state=0)
 
#Running at default
gb_clf = GradientBoostingClassifier(random_state = 0)
gb_clf.fit(X_train, y_train)

#Accuracy
predictions = gb_clf.predict(X_test) 
print(confusion_matrix(y_test, predictions))
accuracy_score(y_test, predictions)

#Finding the optimal parameters
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(random_state = 0,learning_rate = learning_rate)
    gb_clf.fit(X_train, y_train)
    scores= cross_val_score(estimator= gb_clf, X=X_std, y=y, cv=5)
    
    print("Learning rate: ", learning_rate)
    print(learning_rate, '-', numpy.average(scores))
    
    
#0.25 gives us the best learning rate
    
for max_features in range(2,10):
    gb_clf = GradientBoostingClassifier(random_state = 0,max_features = max_features)
    gb_clf.fit(X_train, y_train)
    scores= cross_val_score(estimator= gb_clf, X=X_std, y=y, cv=5)
    
    print("Max_feature ", max_features)
    print(max_features, '-', numpy.average(scores))

# 0.3 is the best max feature    
for max_depth in range(2,10):
    gb_clf = GradientBoostingClassifier(random_state = 0,max_depth = max_depth)
    gb_clf.fit(X_train, y_train)
    scores= cross_val_score(estimator= gb_clf, X=X_std, y=y, cv=5)
    
    print("Max_depth ", max_depth)
    print(max_depth, '-', numpy.average(scores))   

# 2 is optimal depth
#Now accuracy score with optimal parameters
gb_clf_optimal = GradientBoostingClassifier(learning_rate = 0.25,max_features = 3, max_depth = 2, random_state = 0)
gb_clf_optimal.fit(X_train, y_train)

#Accuracy
predictions = gb_clf_optimal.predict(X_test) 
print(confusion_matrix(y_test, predictions))
accuracy_score(y_test, predictions)

       
    