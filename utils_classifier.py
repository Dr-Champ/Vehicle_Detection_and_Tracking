#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:31:23 2017

@author: Champ
"""
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn import svm
from keras.regularizers import l2
import keras.layers as layers
import numpy as np


def create_and_train_svc(x_train, y_train, x_valid, y_valid):
    clf_svc = svm.SVC()
    clf_svc.fit(x_train, y_train)
    clf_svc_acc = accuracy_score(clf_svc.predict(x_valid), y_valid)
    return clf_svc, clf_svc_acc

def create_and_train_lsvc(x_train, y_train, x_valid, y_valid):
    clf_lsvc = svm.LinearSVC()
    clf_lsvc.fit(x_train, y_train)
    clf_lsvc_acc = accuracy_score(clf_lsvc.predict(x_valid), y_valid)
    return clf_lsvc, clf_lsvc_acc

def create_and_train_cnn_1l(x_train, y_train, x_valid, y_valid):
    """
    Create and train a CNN with 32 (3x3) filters, a dropout layer, followed by
    2 FC layers and an output layer. Expected inputs to be list of gray-scaled
    images of shape (32, 32)

    returns:
        model
        history
        accuracy - accuracy on the given validation data
    """
    kernel_depth = 32
    kernel_size = (3, 3)
    drop_prob = 0.5
    fc1 = 200
    fc2 = 50
    l2_weight = 0.1
    batch = 32
    epoch = 15
    model = Sequential()
    model.add(layers.convolutional.Convolution2D(
            kernel_depth, kernel_size[0], kernel_size[1], border_mode="valid",
            input_shape=(32, 32, 1)))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.core.Flatten())
    model.add(layers.core.Dense(
            fc1, init="glorot_normal", W_regularizer=l2(l2_weight),
            activation="relu"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Dense(
            fc2, init="glorot_normal", W_regularizer=l2(l2_weight),
            activation="relu"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Dense(1, activation="sigmoid"))
    model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=batch, nb_epoch=epoch)
    results = model.predict_classes(x_valid, batch_size=batch)
    acc = accuracy_score(results, y_valid)

    return model, history, acc

def create_and_train_cnn_2l(x_train, y_train, x_valid, y_valid):
    """
    Create and train a 2-layer CNN, a dropout layer, followed by
    2 FC layers and an output layer. Expected inputs to be list of gray-scaled
    images of shape (32, 32)

    returns:
        model
        history
        accuracy - accuracy on the given validation data
    """
    kernel1_depth = 32
    kernel1_size = (3, 3)
    kernel2_depth = 64
    kernel2_size = (3, 3)
    drop_prob = 0.5
    fc1 = 200
    fc2 = 50
    l2_weight = 0.1
    batch = 32
    epoch = 15
    model = Sequential()
    model.add(layers.convolutional.Convolution2D(
            kernel1_depth, kernel1_size[0], kernel1_size[1], border_mode="valid",
            input_shape=(32, 32, 1)))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.convolutional.Convolution2D(
            kernel2_depth, kernel2_size[0], kernel2_size[1], border_mode="valid"))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.core.Flatten())
    model.add(layers.core.Dense(
            fc1, init="glorot_normal", W_regularizer=l2(l2_weight),
            activation="relu"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Dense(
            fc2, init="glorot_normal", W_regularizer=l2(l2_weight),
            activation="relu"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Dense(1, activation="sigmoid"))
    model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=batch, nb_epoch=epoch)
    results = model.predict_classes(x_valid, batch_size=batch)
    acc = accuracy_score(results, y_valid)

    return model, history, acc

def create_and_train_cnn_3l(x_train, y_train, x_valid, y_valid):
    """
    Create and train a 3-layer CNN, followed by 1 FC layer and an output layer.
    Expected inputs to be list of gray-scaled images of shape (32, 32)

    returns:
        model
        history
        accuracy - accuracy on the given validation data
    """
    kernel1_depth = 32
    kernel1_size = (3, 3)
    kernel2_depth = 64
    kernel2_size = (3, 3)
    kernel3_depth = 96
    kernel3_size = (3, 3)
    drop_prob = 0.5
    fc1 = 50
    l2_weight = 0.1
    batch = 32
    epoch = 15
    model = Sequential()
    model.add(layers.convolutional.Convolution2D(
            kernel1_depth, kernel1_size[0], kernel1_size[1], border_mode="valid",
            input_shape=(32, 32, 1)))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.convolutional.Convolution2D(
            kernel2_depth, kernel2_size[0], kernel2_size[1], border_mode="valid"))
    model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.convolutional.Convolution2D(
            kernel3_depth, kernel3_size[0], kernel3_size[1], border_mode="valid"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Activation("relu"))
    model.add(layers.core.Flatten())
    model.add(layers.core.Dense(
            fc1, init="glorot_normal", W_regularizer=l2(l2_weight),
            activation="relu"))
    model.add(layers.core.Dropout(drop_prob))
    model.add(layers.core.Dense(1, activation="sigmoid"))
    model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size=batch, nb_epoch=epoch)
    results = model.predict_classes(x_valid, batch_size=batch)
    acc = accuracy_score(results, y_valid)

    return model, history, acc





