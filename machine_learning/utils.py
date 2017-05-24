#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:06:19 2017

@author: Ronald Kemker
"""

import numpy as np
from sklearn.model_selection import train_test_split

#TODO: Need to add documentation

def class_weights(labels, mu , numClasses=None):
    if numClasses is None:
        numClasses= np.max(labels)+1    
    class_weights = np.bincount(labels, minlength=numClasses)       
    class_weights = np.sum(class_weights)/class_weights
    class_weights[np.isinf(class_weights)] = 0
    class_weights = mu * np.log(class_weights)
    class_weights[class_weights < 1] = 1
    return class_weights


def train_test_split_per_class(X, y, train_size=None, test_size=None):
    
    sh = np.array(X.shape)
    
    num_classes = len(np.bincount(y))
    
    sh[0] = 0
    X_train_arr =  np.zeros(sh, dtype=X.dtype)
    X_test_arr = np.zeros(sh, dtype=X.dtype)
    y_train_arr = np.zeros((0), dtype=y.dtype)
    y_test_arr = np.zeros((0), dtype=y.dtype)

    for i in range(num_classes):
        X_train, X_test, y_train, y_test = train_test_split(X[y==i], y[y==i],
                                                            train_size=train_size,
                                                            test_size=test_size)
        
        X_train_arr =  np.append(X_train_arr, X_train, axis=0)
        X_test_arr = np.append(X_test_arr, X_test, axis=0)
        y_train_arr = np.append(y_train_arr, y_train)
        y_test_arr = np.append(y_test_arr, y_test)
        
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr

def set_gpu(device=None):
    import os
    
    if device is None:
        device=""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    