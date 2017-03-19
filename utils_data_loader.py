#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:30:48 2017

@author: Champ
"""
import glob
import matplotlib.pyplot as mplt


def load_files(file_path, images, sampling_rate=0):
    """
    Loads image files and do a sub-sampling of the read files at the given rate.
    This is to reduce duplication of similar scenes from video images.

    params:
        file_path - file pattern for glob
        images - array to append images to
        sampling_rate - read every sampling_rate image to the list
    returns:
        images - a list of read / sampled images
    """
    print("Reading files from path %s" % (file_path))
    paths = glob.glob(file_path)

    if sampling_rate < 1:
        sampling_rate = 1

    for counter in range(len(paths)):
        if counter % sampling_rate == 0:
            image = mplt.imread(paths[counter])

            # in case it's a 4-channel RGBA, strip away the A channel
            if image.shape[2] > 3:
                image = image[:, :, 0:3]

            images.append(image)

    return images

