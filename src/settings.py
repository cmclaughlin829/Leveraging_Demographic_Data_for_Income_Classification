#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Settings Module
"""
# List of all features in original data set
all_features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'earnings']

# Feature to be used as model target
target = 'earnings'

# List of features not included in model
removed_features = ['fnlwgt', 'education', 'earnings']

# List of categorical features
categorical_features = ['workclass', 'marital_status', 'occupation',
                      'relationship', 'race', 'sex', 'native_country']
