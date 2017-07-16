#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Module

Module to train and export a gradient boosting classifier
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from utils import *
from settings import *


if __name__ == "__main__":
    train = read()
    print('Reading Data')

    #Clean the target column
    train = target_clean(train, target)

    #Clean the feature columns that contain strings
    train = string_feature_clean(train, categorical_features, '?')

    #Separate the predictors and targets
    X_train, y_train = data_split(train, removed_features, target)

    #Convert categorical features to binary features
    X_train = pd.get_dummies(X_train, columns=categorical_features)

    #Create sample weight to account for class imbalance
    y_df = pd.DataFrame(y_train)
    y_df['sample_weight'] = np.where(y_df['>50K']==0, 1, 3)

    predictor=GradientBoostingClassifier(max_depth=2, learning_rate=0.1,
                                         n_estimators=1500)

    #Fit model to weighted training data
    print('Fitting model')
    predictor.fit(X_train, y_train, sample_weight=y_df['sample_weight'])

    #Export fit model
    print('Model Successfully Fit')
    filename = 'fit_gbc_model.sav'
    pickle.dump(predictor, open(filename, 'wb'))
    print('Fit Model Exported to ', os.path.dirname(os.path.realpath(__file__)))
    sys.exit()
