#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction Module

Module to provide predictions of test data utilizing model that has been
fit and imported
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from utils import *
from settings import *

if __name__ == "__main__":
    test = read()
    print('Reading Data')

    #Clean the target column
    test = target_clean(test, target)

    #Clean the feature columns that contain strings
    test = string_feature_clean(test, categorical_features, '?')

    #Separate the predictors and targets
    X_test, y_test = data_split(test, removed_features, target)

    #Convert categorical features to binary features
    X_test = pd.get_dummies(X_test, columns=categorical_features)

    #Add column of zeros for feature value that was not contained in test set
    X_test['native_country_Holand-Netherlands']=0

    filename = 'fit_gbc_model.sav'
    predictor = pickle.load(open(filename, 'rb'))
    predicted_values = predictor.predict(X_test)
    print('Test Data Predictions Obtained')

    #Obtain model metrics
    mat = confusion_matrix(y_test, predicted_values)
    probs = predictor.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    x = input('Would you like to view the Classification Report? (y/n) ').lower()
    if x not in ('y', 'n'):
        print('Please try again.')
    elif x == 'y':
        print (classification_report(y_test, predicted_values))
    elif x == 'n':
        pass

    x = input('Would you like to view the Confusion Matrix? (y/n) ').lower()
    if x not in ('y', 'n'):
        print('Please try again.')
    elif x == 'y':
        print (mat)
    elif x == 'n':
        pass

    x = input('Would you like to save a copy of the Confusion Matrix? (y/n) ').lower()
    if x not in ('y', 'n'):
        print('Please try again.')
    elif x == 'y':
        print ('Saving image to ',
                os.path.dirname(os.path.realpath(__file__)))
        p_confusion_matrix(mat)
    elif x == 'n':
        pass

    x = input('Would you like to save a copy of the ROC Curve? (y/n) ').lower()
    if x not in ('y', 'n'):
        print('Please try again.')
    elif x == 'y':
        print ('Saving image to ',
                os.path.dirname(os.path.realpath(__file__)))
        p_roc(fpr, tpr, roc_auc)
    sys.exit()
