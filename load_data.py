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

    elif which_data == 'nus-wide21':
        img_rows = img_cols = 64
        img_channels = 3
        num_classes = 21
        (x_train, y_train), (x_test, y_test) = nus_wide21().load_data()
    elif which_data == 'ut-zap50k':
        img_rows = 32
        img_cols = 32
        num_classes = 4
        (x_train, y_train), (x_test, y_test) = UT_ZAP50K().load_data()
        x_train = 255 - x_train
        x_test = 255 - x_test
    elif which_data == 'svhn':
        img_rows = 32
        img_cols = 32
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = SVHN().load_data()

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


class nus_wide21:

    def __init__(self):
        self.data_npy = 'nus-wide_data.npy'
        self.label_npy = 'nus-wide_label.npy'
        self.image_path = '/home/tsl/test/dataset/NUS-WIDE21/nuswide-21/'  # no useful temporarily
        self.x_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.rows = 64
        self.cols = 64
        self.channels = 3
        self.TestNum = 2100


    def load_data(self):

        # Gernerate nuw-wide npy format
        # ImageNum = 0
        # LastLabelName = ''
        # LabelIndex = 0
        # ImageIndex = 0
        # LabelNum = 0
        # for root, dirs, files in os.walk(self.image_path):
        #     ImageNum += len(files)
        # print ImageNum
        # x_data = np.zeros((ImageNum, self.rows, self.cols, self.channels))
        # y_label = np.zeros(ImageNum)
        # for root, dirs, files in os.walk(self.image_path):
        #     if len(files) == 0:
        #         continue
        #
        #     for i in range(len(files)):
        #
        #         if i % 100 == 0:
        #             print 'Label=%d/%d, Image=%d/%d' % (LabelIndex, LabelNum, i, len(files))
        #
        #         img_path = root + '/' + files[i]
        #         im = np.array(Image.open(img_path))
        #         x_data[ImageIndex, :, :, :] = im
        #
        #         y_label[ImageIndex] = LabelIndex
        #
        #         ImageIndex += 1
        #
        #     LabelIndex += 1

        # np.save('nus-wide_train_data.npy', x_data)
        # np.save('nus-wide_train_label.npy', y_label)

        x_data = np.load(self.data_npy)
        y_data = np.load(self.label_npy)

        ImageNum = x_data.shape[0]

        indexes = range(x_data.shape[0])
        random.shuffle(indexes)

        self.x_train = x_data[0:ImageNum-self.TestNum, :, :, :]
        self.y_train = y_data[0:ImageNum-self.TestNum]
        self.x_test = x_data[ImageNum-self.TestNum:ImageNum]
        self.y_test = y_data[ImageNum-self.TestNum:ImageNum]

        return (self.x_train, self.y_train), (self.x_test, self.y_test)


class UT_ZAP50K:
    def __init__(self):
        self.x_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.test_num = 1000
        self.rows = 96
        self.cols = 128
        self.channels = 3
        self.classes = 4
        self.ImageDir = "/home/tsl/test/dataset/ut-zap50k/"

    def load_data(self):
        # ImageNum = 0
        # ImageIndex = 0
        # LabelIndex = 0
        # Labels = ['Boots', 'Sandals', 'Shoes', 'Slippers']
        #
        # for root, dirs, files in os.walk(self.ImageDir):
        #     ImageNum += len(files)
        #
        # train_num = ImageNum-self.classes*self.test_num
        # test_num = ImageNum - train_num
        #
        # self.x_train = np.zeros((train_num, self.rows, self.cols, self.channels), np.uint8)
        # self.y_train = np.zeros((train_num, 1), np.uint8)
        # self.x_test = np.zeros((test_num, self.rows, self.cols, self.channels), np.uint8)
        # self.y_test = np.zeros((test_num, 1), np.uint8)
        #
        # train_index = 0
        # test_index = 0
        # for labels in Labels:
        #     class_index = 0
        #     for root, dirs, files in os.walk(self.ImageDir+labels):
        #         if len(files) == 0:
        #             continue
        #
        #         for filename in files:
        #             if ImageIndex % 100 == 0:
        #                 print 'Label=%d/%d, Image=%d/%d' % (LabelIndex, 4, ImageIndex, ImageNum)
        #             img = Image.open(root+'/'+filename)
        #             img = img.resize((self.cols, self.rows), Image.ANTIALIAS)
        #             if img.mode != 'RGB':
        #                 img = img.convert('RGB')
        #             img = np.array(img, np.uint8)
        #             if class_index < self.test_num:
        #                 self.x_test[test_index, :, :, :] = img
        #                 self.y_test[test_index] = LabelIndex
        #                 test_index += 1
        #             else:
        #                 self.x_train[train_index, :, :, :] = img
        #                 self.y_train[train_index] = LabelIndex
        #                 train_index += 1
        #             ImageIndex += 1
        #             class_index += 1
        #     LabelIndex += 1
        #
        # np.save('UT-ZAP50K_train_data.npy', self.x_train)
        # np.save('UT-ZAP50K_train_label.npy', self.y_train)
        # np.save('UT-ZAP50K_test_data.npy', self.x_test)
        # np.save('UT-ZAP50K_test_label.npy', self.y_test)

        print 'load UT-ZAP50K files...\n'
        self.x_train = np.load('/opt/Data/tsl/dataset/UT-ZAP50K/UT-ZAP50K_train_data_32x32.npy')
        self.y_train = np.load('/opt/Data/tsl/dataset/UT-ZAP50K/UT-ZAP50K_train_label_32x32.npy')
        self.x_test = np.load('/opt/Data/tsl/dataset/UT-ZAP50K/UT-ZAP50K_test_data_32x32.npy')
        self.y_test = np.load('/opt/Data/tsl/dataset/UT-ZAP50K/UT-ZAP50K_test_label_32x32.npy')

        return (self.x_train, self.y_train), (self.x_test, self.y_test)


class SVHN:
    def __init__(self):
        self.x_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.test_num = 1000
        self.rows = 32
        self.cols = 32
        self.channels = 3
        self.classes = 4
        self.trainmat = "/opt/Data/tsl/dataset/SVHN/train_32x32_channel_last.mat"
        self.testmat = "/opt/Data/tsl/dataset/SVHN/test_32x32_channel_last.mat"

    def load_data(self):
        train_data = sio.loadmat(self.trainmat)
        test_data = sio.loadmat(self.testmat)

        self.x_train = train_data['X']
        self.y_train = train_data['y']-1
        self.x_test = test_data['X']
        self.y_test = test_data['y']-1

        return (self.x_train, self.y_train), (self.x_test, self.y_test)



SVHN().load_data()
