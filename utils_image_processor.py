#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:30:25 2017

@author: Champ
"""

import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as mplt
from scipy.ndimage.measurements import label

car_center_diviation_threshold = 60
image_size = None

# sliding window parameters
X_START_STOP = (None, None)
Y_START_STOP = (470, 610)
NEAR_WINDOW=(280, 280)
FAR_WINDOW=(90, 90)
X_OVERLAP = 0.9
Y_STEPS=9

# Canny edge detection parameters
BLUR_KERNEL = (5, 5)
CANNY_LOW = 100
CANNY_HIGH = 200

# For svc, threshold / frame = 10-17
# For cnn, threshold / frame = ?

classifier_cnn = None
classifier_svc = None
data_scaler = None
use_cnn = True
historical_threshold = 6
heatmap_threshold_per_frame = 15
heatmap_records = []
car_records = []
CAR_BBOX_MIN_MAX = (90, 280)

frame_count = 1
frame_count_plot_fig = 200

class Car:

    def __init__(self, loc_x=None, loc_y=None):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self._new_centers = []

    def add_new_center(self, center):
        self._new_centers.append(center)

    def has_new_center(self):
        return len(self._new_centers) > 0

    def compute_new_center(self):
        if len(self._new_centers) > 0:
            tot_x = 0
            tot_y = 0
            for center in self._new_centers:
                tot_x = tot_x + center[1]
                tot_y = tot_y + center[0]
            avg_new_center = (
                    tot_y / len(self._new_centers), tot_x / len(self._new_centers))
            self._new_centers = []
            if self.loc_x == None and self.loc_y == None:
                self.loc_x = avg_new_center[1]
                self.loc_y = avg_new_center[0]
            else:
                self.loc_x = (self.loc_x + avg_new_center[1]) / 2
                self.loc_y = (self.loc_y + avg_new_center[0]) / 2

def flip_lr(images):
    new_images = []
    for image in images:
        new_images.append(image)
        new_images.append(np.fliplr(image))

    return new_images

def resize_images(images, shape):
    """
    Resizes all images to the given shape (h, w) and return a new list
    containing resized images
    """
    resized_images = []
    for image in images:
        resized_images.append(cv2.resize(image, shape))
    return resized_images

def convert_color_space(images, dst_color_space):
    """
    Converts given images into the desired color space and return a new list
    of converted images
    """
    converted_images = []
    for image in images:
        converted_images.append(cv2.cvtColor(image, dst_color_space))
    return converted_images

def normalize_gray_images(images):
    """
    Normalize gray images to 0-1 if necessary
    """
    # check the max value of the first image. Not perfect, but should do for now.
    norm_images = []
    if images[0].max() > 1:
        for image in images:
            image = image / 255
            norm_images.append(image)
    else:
        norm_images = images
    return norm_images

def compute_canny_edges(images):
    """
    Compute canny edges from images array and return a new array

    params:
        images - list containing gray images (scaled to 0-1)
    """
    canny_images = np.zeros_like(images)
    for index in range(len(images)):
        image = images[index]
        image_blur = (255 * cv2.GaussianBlur(image, BLUR_KERNEL, 0)).astype(np.uint8)
        image_canny = cv2.Canny(image_blur, CANNY_LOW, CANNY_HIGH)
        canny_images[index, :, :] = image_canny

    return canny_images


def compute_hog(images, orientations=12, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), plot_fig=False):
    """
    Compute HOG features for all images. The resulting feature vector will be
    of size (# blocks in x) * (# blocks in y) * (cells per block in x) *
    (cells per block in y) * orientations

    params:
        images - list of gray-scaled images
    """
    hog_features = []
    for image in images:
        hog_feature = hog(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualise=False,
                transform_sqrt=False,
                feature_vector=True,
                normalise=None)
        hog_features.append(hog_feature)

    if plot_fig:
        _, hog_img = hog(
                images[0],
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualise=True,
                transform_sqrt=False,
                feature_vector=True,
                normalise=None)
        mplt.figure()
        mplt.imshow(hog_img)

    return hog_features

def compute_HLS_histogram_and_binning(images, bins_range, nbins=32):
    """
    Compute the histogram of the HLS color channels and its spatial
    binning of the given HLS images
    """
    feature_list = []
    for image in images:
        image_hist_H = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
#        image_hist_S = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
#        image_spatial_bin = image[:, :, 0].ravel()
#        image_feature = np.concatenate((image_hist_H[0], image_hist_S[0], image_spatial_bin))
#        image_feature = np.concatenate((image_hist_H[0], image_hist_S[0]))
        image_feature = image_hist_H[0]
        feature_list.append(image_feature)

    bin_centers = (image_hist_H[1][1:] + image_hist_H[1][0:(len(image_hist_H) - 1)]) / 2
    return feature_list, bin_centers

def combine_features(features1, features2):
    """
    Combine features from features 1 and 2 element wise
    """
    combined_features = []
    for index in range(len(features1)):
        combined_feature = np.concatenate((features1[index], features2[index]))
        combined_features.append(combined_feature)

    return combined_features

def compute_sliding_window(img, x_start_stop=(None, None),
        y_start_stop=(None, None), near_window=(400, 400), far_window=(20, 20),
        x_overlap=0.5, y_steps=10):
    """
    Compute bounding boxes on the given image to search for cars. Note that if
    start / stop positions for x and y are not given, they will be set to maximum
    values applicable to the given image.

    params:
        near_window - (x, y) of the near-side window (window's bottom at y_stop)
        far_window - (x, y) of the far-side window (window's bottom at y_start)
        x_overlap - overlapping ratio along x axis for adjacent boxes
        y_steps - total steps in y direction
    """
    h = img.shape[0]
    w = img.shape[1]

    if x_start_stop == (None, None):
        x_start_stop = (0, w)

    if y_start_stop == (None, None):
        y_start_stop = (far_window[1], h)

    if y_start_stop[0] < far_window[1]:
        y_start_stop[0] = far_window[1]

    # Initialize a list to append window positions to
    window_list = []

    # Compute the span of the region to be searched
    x_search_range = x_start_stop[1] - x_start_stop[0]
    y_search_range = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in y
    y_step_size = round(y_search_range / (y_steps - 1))
    window_width_step_size = round((near_window[0] - far_window[0]) / (y_steps - 1))
    window_height_step_size = round((near_window[1] - far_window[1]) / (y_steps - 1))

    for y_step in range(y_steps):
        y_bottom = y_start_stop[0] + (y_step * y_step_size)
        window_width = far_window[0] + (y_step * window_width_step_size)
        window_height = far_window[1] + (y_step * window_height_step_size)
        y_top = y_bottom - window_height

        # Compute the number of pixels per step in x
        x_step_size = window_width - round(x_overlap * window_width)

        # Compute the total number of windows in x axis. Use ceil() to include
        # windows that might be off the right boundary of the image too
        x_steps = int(np.ceil((x_search_range - window_width) / x_step_size)) + 1

        for x_step in range(x_steps):
            x_left = x_start_stop[0] + (x_step * x_step_size)
            x_right = x_left + window_width

            corner1 = (x_left, y_top) # top left
            corner2 = (x_right, y_bottom) # bottom right
            window = (corner1, corner2)
            window_list.append(window)

    return window_list

def draw_boxes(img, bboxes, y_start_stop=(None, None), thick=6):
    """
    Draws boxes on a copy of the given image and return it. Colors of the boxes
    vary depending on their locations on the y axis.
    """
    imcopy = np.copy(img)
    max_color = 255
    h = img.shape[0]

    if y_start_stop == (None, None):
        y_start_stop = (0, h)

    y_range = y_start_stop[1] - y_start_stop[0]

    for bbox in bboxes:
        value = round(max_color * (bbox[1][1] - y_start_stop[0]) / y_range)
#        value = max(value, 10)  # prevent it from being total black
        color = (0, 0, value)
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def extract_images_from_search_boxes(image, bbox, image_size,
        converted_to_gray=True):
    """
    params:
        image - a RGB image to extract bbox from
        converted_to_gray - True to convert images to gray scale
    """
    image_boxes = []

    for box in bbox:
        x_min = box[0][0]
        y_min = box[0][1]
        x_max = box[1][0]
        y_max = box[1][1]
        image_box = image[y_min: y_max, x_min: x_max, :]
        image_boxes.append(image_box)

    image_boxes = resize_images(image_boxes, image_size)
    if converted_to_gray:
        return convert_color_space(image_boxes, cv2.COLOR_RGB2GRAY)
    else:
        return image_boxes

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def draw_car_bboxes(img, cars, y_start_stop, bbox_min_max):
    slope = (bbox_min_max[1] - bbox_min_max[0]) / (y_start_stop[1] - y_start_stop[0])

    for car in cars:
        bbox_size = slope * (car.loc_y - y_start_stop[0]) + bbox_min_max[0]
        bbox = ((int(car.loc_x - (bbox_size / 2)), int(car.loc_y - (bbox_size / 2))),
                (int(car.loc_x + (bbox_size / 2)), int(car.loc_y + (bbox_size / 2))))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img

def compute_average_label_centers(labels):
    """
    Get image of labels, compute and return each one's center
    """
    centers = []
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        center_y = np.average(nonzero[0])
        center_x = np.average(nonzero[1])
        center = (center_y, center_x)
        centers.append(center)

    return centers

def compute_euclidian_distance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))

def identify_cars_from_centers(cars, centers):
    """
    Each center will be considered as representing the same car only if there
    was a car "nearby" from previous frame. If the center represents the same
    car, modify the car object to be at the new average center. If the center
    represents a new car, create a new Car object and add to the list.

    Note that any existing car in the cars list which does not have corresponding
    center in centers will be removed.
    """
    # Check if each new center belongs to any existing car or a new car
    for center in centers:
        assigned_to_car = False
        for car in cars:
            distance = compute_euclidian_distance(
                    (center[0], center[1]), (car.loc_y, car.loc_x))
            if distance < car_center_diviation_threshold:
                car.add_new_center(center)
                assigned_to_car = True
                break

        if not assigned_to_car:
            new_car = Car()
            new_car.add_new_center(center)
            new_car.compute_new_center() # not strictly correct, but to have x/y locations for the next iteration
            cars.append(new_car)

    # Now update each car's location. If the car doesn't have any newly added
    # center then it is removed from the new list
    updated_cars = []
    for car in cars:
        if car.has_new_center():
            car.compute_new_center()
            updated_cars.append(car)

    return updated_cars

def detect_cars_in_image(image, bboxes, image_size, clf_cnn, clf_svm,
        x_train_scaler, heat_maps, cars, use_cnn, plot_fig=False):
    """
    - detect car in search boxes
    - build heatmap (from the given historical heatmap). Remove oldest map away.
    - apply threshold to heatmap to remove false positives
    - compute average centers of the detected labels. average their positions
      from history.
    - draw boxes on the detected cars

    params:
        image_size - size of images to feed to the classifier
        clf_cnn - a CNN classifier trained to detect a car from gray image
        clf_svm - a SVM classifier trained to detect a car from gray image
        x_train_scaler - HOG feature scaler, trained for test data set
        heat_maps - a list of previous heat maps to merge with this one and
            compute threshold over them
        cars - a list of cars from the previous frame
        use_cnn - True to use CNN classifier, False to use SVM
    """
    # Get a list of resized images of size 32x32 from bboxes
    image_boxes = extract_images_from_search_boxes(image, bboxes, image_size, False)
    images_hls = convert_color_space(image_boxes, cv2.COLOR_RGB2HLS)
    images_gray = convert_color_space(image_boxes, cv2.COLOR_RGB2GRAY)
    images_gray = normalize_gray_images(images_gray)

    if use_cnn and clf_cnn is not None:
        images_gray = np.array(images_gray)
        images_gray = images_gray[:, :, :, np.newaxis]
        results = clf_cnn.predict_classes(images_gray)
    elif not use_cnn and clf_svm is not None:
        hogs = compute_hog(images_gray, plot_fig=False)
        H_feature, _ = compute_HLS_histogram_and_binning(images_hls, (0, 360))
        features = combine_features(hogs, H_feature)
        features = x_train_scaler.transform(features)
        results = clf_svm.predict(features)

    image_heat_map = np.zeros_like(image[:, :, 0])

    for index in range(len(bboxes)):
        if results[index] == 1:
            # positive detection. Add to heat map
            box = bboxes[index]
            image_heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # combining previous heatmaps and this one together to apply threshold
    if len(heat_maps) >= historical_threshold:
        heat_maps.pop(0)

    heat_maps.append(image_heat_map)
    combined_heatmap = np.zeros_like(image_heat_map)

    for heat_map in heat_maps:
        # TODO should we apply frame threshold here ?
        truth_mat = heat_map > 0
        combined_heatmap[truth_mat] += heat_map[truth_mat]

    # TODO then apply only historical threshold here ?
    combined_heatmap[combined_heatmap < (heatmap_threshold_per_frame * historical_threshold)] = 0

    # assign labels, compute average centers of the detected cars
    labels = label(combined_heatmap)
    image_cars = draw_labeled_bboxes(np.copy(image), labels)
    print("Detected %i cars" % (labels[1]))

#    centers = compute_average_label_centers(labels)
#    cars = identify_cars_from_centers(cars, centers)
#
#    # draw boxes on cars in the current frame with pre-defined box size
#    # scaled toward the horizon
#    image_cars = draw_car_bboxes(np.copy(image), cars,
#            y_start_stop=Y_START_STOP, bbox_min_max=CAR_BBOX_MIN_MAX)
#    print("Detected %i cars" % (len(cars)))

    if plot_fig:
#        mplt.figure()
#        mplt.subplot(121)
#        mplt.title("Original")
#        mplt.imshow(image)
#        mplt.subplot(122)
#        mplt.title("Current heat map")
#        mplt.imshow(image_heat_map, cmap="gray")

        mplt.figure()
        mplt.subplot(121)
#        mplt.title("Combined heat map")
#        mplt.imshow(combined_heatmap, cmap="gray")
        mplt.title("Labelled combined heat map")
        mplt.imshow(labels[0])
        mplt.subplot(122)
        mplt.title("Resulting detection")
        mplt.imshow(image_cars)
        mplt.show()

#        mplt.figure()
#        mplt.title("Final detection")
#        mplt.imshow(image_cars)

    return image_cars

def detect_cars_in_video_frame(image, plot_fig=False):
    """
    Just a wrapper to detect_cars_in_image()

    Note: It uses heavily global variables from external trained models and other
    configurations. It's ugly, but we'll get us by for now.
    """
    bboxes = compute_sliding_window(
            image, x_start_stop=X_START_STOP, y_start_stop=Y_START_STOP,
            near_window=NEAR_WINDOW, far_window=FAR_WINDOW, x_overlap=X_OVERLAP,
            y_steps=Y_STEPS)

    if plot_fig:
        image_boxes = draw_boxes(image, bboxes, y_start_stop=Y_START_STOP)
        mplt.figure()
        mplt.title("Search boxes")
        mplt.imshow(image_boxes)

    image_cars = detect_cars_in_image(
            image, bboxes, image_size, classifier_cnn, classifier_svc,
            data_scaler, heatmap_records, car_records, use_cnn, True)

    return image_cars






