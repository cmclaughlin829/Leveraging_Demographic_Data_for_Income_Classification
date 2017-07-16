#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from settings import *

def read():
    """
    Function to read in data sets
    """
    x = input('enter file name ')
    data = pd.read_csv("../data/{}".format(x), names=all_features, index_col=False)
    return data

def target_clean(df, target):
    """
    Function to remove unnecessary characters
    from string contained in target column
    """
    if df[target][0][0]==' ':
        df[target]=df[target].map(lambda x: str(x)[1:])
    if df[target][0][-1]=='.':
        df[target]=df[target].map(lambda x: str(x)[:-1])
    return df

def string_feature_clean(df, column_list, null_value):
    """
    Function to clean string features
    """
    for column in column_list:
        if df[column][0][0]==' ':
            df[column] = df[column].map(lambda x: str(x)[1:])
        else:
            pass
    for column in column_list:
        df = df[df[column] != null_value]
    df = df.reset_index(drop=True)
    return df

def data_split(data_in, removed_features, target):
    """
    Function to split features and targets
    """
    X_out = data_in.drop(data_in[removed_features], axis=1)
    y_out = pd.get_dummies(data_in[target], columns=target)['>50K']
    return X_out, y_out

def p_confusion_matrix(matrix):
    """
    Function to create and save a confusion matrix
    """
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size=18)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    fig.savefig('boosting_test_matrix.png', transparent = False, dpi = 80, bbox_inches = 'tight')

def p_roc(fpr, tpr, roc_auc):
    """
    Function to create and save a ROC Curve
    """
    fig, ax = plt.subplots(figsize = (10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig.savefig('boosting_test_roc.png', transparent = False, dpi = 80, bbox_inches = 'tight')
