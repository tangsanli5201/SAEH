# ================================================================================================
# Code for paper 'Deep Supervised Auto-encoder Hashing for Image Retrieval'.
# Currently, you may not use this file for any purpose unless you got permission from the author.
# Author: Sanli Tang
# Email: tangsanli@sjtu.edu.cn
# Organization: Shanghai Jiaotong Univ.
# Modified Time: 2017/10/25
# ================================================================================================

import keras
import numpy as np
from keras.datasets import cifar10, mnist
import scipy.io as sio
from PIL import Image
import os
import random
import scipy.io as sio


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test


def load_data(which_data, one_hot=True):
    img_rows = img_cols = 0
    img_channels = 0
    num_classes = 0
    if which_data == 'mnist':
        img_rows = img_cols = 28
        img_channels = 1
        num_classes = 10
        # load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    elif which_data == 'cifar10':
        img_rows = img_cols = 32
        img_channels = 3
        num_classes = 10
        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif which_data == 'ut-zap50k':
        print 'you should define your own loading method for UT-Zap50K dataset'
        # you should define your own loading method for UT-Zap50K dataset
        # img_rows = 32
        # img_cols = 32
        # num_classes = 4
        # (x_train, y_train), (x_test, y_test) = UT_ZAP50K().load_data()
        # x_train = 255 - x_train
        # x_test = 255 - x_test
    elif which_data == 'svhn':
        print 'you should define your own loading method for SVHN dataset'
        # you should define your own loading method for SVHN dataset
        # img_rows = 32
        # img_cols = 32
        # num_classes = 10
        # (x_train, y_train), (x_test, y_test) = SVHN().load_data()

    if one_hot:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    if img_channels == 1:
        num_train_sample = len(x_train)
        num_test_sample = len(x_test)
        x_train = np.reshape(x_train, [num_train_sample, img_rows, img_cols, img_channels])
        x_test = np.reshape(x_test, [num_test_sample, img_rows, img_cols, img_channels])

    return (x_train, y_train), (x_test, y_test)
