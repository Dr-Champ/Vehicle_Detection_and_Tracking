#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:46:45 2017

@author: Champ
"""

import utils_image_processor as uip
import utils_data_loader as udl
import utils_classifier as ucf
import matplotlib.pyplot as mplt
import numpy as np
import keras
import pickle
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip

# FIXME
# Try with 48x48 image. See if CNN detects black car

# ----- Configurable params for model training ----

READ_PICKLE = True
PLOT_FIG = False
RETRAIN_CNN = False
TRAIN_CNN_WITH_GRAY = True # if False, will use Canny edges
PICKLE_DATA = "images_pickle.p"
IMAGE_SIZE = (32, 32)
VALIDATION_RATIO = 0.2
CNN3_MODEL = "cnn3_model.h5"

# ---------------- Training phase -----------------

## Load data
#if READ_PICKLE:
#    with open(PICKLE_DATA, "rb") as f:
#        data = pickle.load(f)
#        cars = data[0]
#        non_cars = data[1]
#else:
#    cars= []
#    udl.load_files("data/vehicles/GTI_Far/*.png", cars, sampling_rate=6)
#    udl.load_files("data/vehicles/GTI_Left/*.png", cars, sampling_rate=6)
#    udl.load_files("data/vehicles/GTI_MiddleClose/*.png", cars, sampling_rate=6)
#    udl.load_files("data/vehicles/GTI_Right/*.png", cars, sampling_rate=6)
#    udl.load_files("data/vehicles/KITTI_extracted/*.png", cars, sampling_rate=0)
#
#    non_cars = []
#    udl.load_files("data/non-vehicles/GTI/*.png", non_cars, sampling_rate=0)
#    udl.load_files("data/non-vehicles/Extras/*.png", non_cars, sampling_rate=0)
#
#    with open(PICKLE_DATA, "wb") as f:
#        pickle.dump([cars, non_cars], f)
#
#print("car samples %i, non-car samples %i" % (len(cars), len(non_cars)))
#
## Rescale value ranges and size
##cars = np.multiply(cars, 255)
##non_cars = np.multiply(non_cars, 255)
#print("car image, max %f, min %f, shape %s"
#      % (cars[0].max(), cars[0].min(), str(cars[0].shape)))
#print("non-car image, max %f, min %f, shape %s"
#      % (non_cars[0].max(), non_cars[0].min(), str(non_cars[0].shape)))
#
#if PLOT_FIG:
#    mplt.figure()
#    mplt.subplot(121)
#    mplt.title("Car")
#    mplt.imshow(cars[0])
#    mplt.subplot(122)
#    mplt.title("Non-car")
#    mplt.imshow(non_cars[0])
#    mplt.show()
#
## Resize images
#print("Resizing and converting images")
#cars = uip.resize_images(cars, IMAGE_SIZE)
#cars_augmented = uip.flip_lr(cars)
#cars_hls = uip.convert_color_space(cars_augmented, cv2.COLOR_RGB2HLS)
#cars_gray = uip.convert_color_space(cars_augmented, cv2.COLOR_RGB2GRAY)
#cars_gray = uip.normalize_gray_images(cars_gray)
#cars_canny = uip.compute_canny_edges(cars_gray)
#
#non_cars = uip.resize_images(non_cars, IMAGE_SIZE)
#non_cars_augmented = uip.flip_lr(non_cars)
##non_cars_augmented = non_cars
#non_cars_hls = uip.convert_color_space(non_cars_augmented, cv2.COLOR_RGB2HLS)
#non_cars_gray = uip.convert_color_space(non_cars_augmented, cv2.COLOR_RGB2GRAY)
#non_cars_gray = uip.normalize_gray_images(non_cars_gray)
#non_cars_canny = uip.compute_canny_edges(non_cars_gray)
#
#print("car samples %i, non-car samples %i"
#      % (len(cars_augmented), len(non_cars_augmented)))
#
#if PLOT_FIG:
#    mplt.figure()
#    mplt.subplot(121)
#    mplt.title("Car")
#    mplt.imshow(cars_gray[0], cmap="gray")
#    mplt.subplot(122)
#    mplt.title("Non-car")
#    mplt.imshow(non_cars_gray[0], cmap="gray")
#    mplt.show()
#
#    mplt.figure()
#    mplt.subplot(121)
#    mplt.title("Car Canny")
#    mplt.imshow(cars_canny[0], cmap="gray")
#    mplt.subplot(122)
#    mplt.title("Non-car Canny")
#    mplt.imshow(non_cars_canny[0], cmap="gray")
#    mplt.show()
#
#    mplt.figure()
#    mplt.subplot(231)
#    mplt.title("Car H")
#    mplt.imshow(cars_hls[0][:, :, 0], cmap="gray")
#    mplt.subplot(232)
#    mplt.title("Car L")
#    mplt.imshow(cars_hls[0][:, :, 1], cmap="gray")
#    mplt.subplot(233)
#    mplt.title("Car S")
#    mplt.imshow(cars_hls[0][:, :, 2], cmap="gray")
#    mplt.subplot(234)
#    mplt.title("Non-car H")
#    mplt.imshow(non_cars_hls[0][:, :, 0], cmap="gray")
#    mplt.subplot(235)
#    mplt.title("Non-car L")
#    mplt.imshow(non_cars_hls[0][:, :, 1], cmap="gray")
#    mplt.subplot(236)
#    mplt.title("Non-car S")
#    mplt.imshow(non_cars_hls[0][:, :, 2], cmap="gray")
#
## Convert images to HOG feature vectors from gray images
#print("Extracting HOG features")
#cars_hog = uip.compute_hog(cars_gray, plot_fig=PLOT_FIG)
#non_cars_hog = uip.compute_hog(non_cars_gray, plot_fig=PLOT_FIG)
#
## Append HLS color histogram features to the HOG features
#print("Extracting and appending H histogram features")
#cars_hist_bin, _ = uip.compute_HLS_histogram_and_binning(cars_hls, (0, 360))
#cars_features = uip.combine_features(cars_hog, cars_hist_bin)
#
#non_cars_hist_bin, _ = uip.compute_HLS_histogram_and_binning(non_cars_hls, (0, 360))
#non_cars_features = uip.combine_features(non_cars_hog, non_cars_hist_bin)
#
## Prepare data and labels. Then shuffle and split data into train / test sets
#print("Split training / validation sets")
#x_train = np.concatenate((cars_features, non_cars_features))
#y_train = np.concatenate((np.ones(len(cars_gray)), np.zeros(len(non_cars_gray))))
#x_train, x_valid, y_train, y_valid = train_test_split(
#        x_train, y_train, test_size=VALIDATION_RATIO)
#
## Standardized HOG, hist data (using properties of training set and apply to
## test set too)
#print("Standardising datasets")
#x_train_scaler = StandardScaler().fit(x_train)
#x_train = x_train_scaler.transform(x_train)
#x_valid = x_train_scaler.transform(x_valid)
#
## Train SVM classifiers
#print("Training predictors")
## 0.982 (HOG), 0.990 (HOG, H hist), 0.992 (HOG, H hist, spatial binning)
#clf_svc, clf_svc_acc = ucf.create_and_train_svc(
#        x_train, y_train, x_valid, y_valid)
#
## 0.954 (HOG), 0.967 (HOG, H hist), 0.962 (HOG, H hist, spatial binning)
#clf_lsvc, clf_lsvc_acc = ucf.create_and_train_lsvc(
#        x_train, y_train, x_valid, y_valid)
#
## Measure and compare accuracies
#print("clf_svc_acc %f, clf_lsvc_acc %f" % (clf_svc_acc, clf_lsvc_acc))
#
## Train CNN (with gray-scaled images)
#if RETRAIN_CNN:
#    if TRAIN_CNN_WITH_GRAY:
#        # Create data sets for gray images and expand dimension of these images
#        # (to be compatable with Keras)
#        x_img_train = np.concatenate((cars_gray, non_cars_gray))
#    else:
#        x_img_train = np.concatenate((cars_canny, non_cars_canny))
#
#    y_img_train = np.concatenate(
#            (np.ones(len(cars_gray)), np.zeros(len(non_cars_gray))))
#    x_img_train = x_img_train[:, :, :, np.newaxis]
#    x_img_train, x_img_valid, y_img_train, y_img_valid = train_test_split(
#            x_img_train, y_img_train, test_size=VALIDATION_RATIO)
#
#    clf_cnn3, history3, clf_cnn3_acc = ucf.create_and_train_cnn_3l(
#            x_img_train, y_img_train, x_img_valid, y_img_valid) # 0.9859
#    clf_cnn3.save(CNN3_MODEL)
#    print("clf_cnn3_acc %f" % (clf_cnn3_acc))
#else:
##    clf_cnn1 = keras.models.load_model(CNN1_MODEL)
##    clf_cnn2 = keras.models.load_model(CNN2_MODEL)
#    clf_cnn3 = keras.models.load_model(CNN3_MODEL)

# -- Configurable parameters for video processing --

USE_TEST_IMAGE = False
TEST_IMAGE = "data/test_images/test4.jpg"
TEST_VIDEO = "test_video.mp4"
#TEST_VIDEO = "project_video.mp4"
OUTPUT_VIDEO = "result_video_6frames_demo.mp4"
uip.classifier_cnn = clf_cnn3
uip.classifier_svc = clf_svc
uip.data_scaler = x_train_scaler
uip.image_size = IMAGE_SIZE
uip.use_cnn = False
uip.heatmap_records = []
uip.car_records = []

# --------------- Processing video -----------------

if USE_TEST_IMAGE:
    image = mplt.imread(TEST_IMAGE)
    image_cars = uip.detect_cars_in_video_frame(image, PLOT_FIG)

    if PLOT_FIG:
        mplt.figure()
        mplt.imshow(image_cars)
else:
    # read video frames and process one by one
    clip = VideoFileClip(TEST_VIDEO)
    overlay_clip = clip.fl_image(uip.detect_cars_in_video_frame)
    overlay_clip.write_videofile(OUTPUT_VIDEO, audio=False)






















