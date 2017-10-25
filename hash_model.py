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
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose, MaxPooling2D
from keras.initializers import he_normal
from keras import regularizers
from keras import objectives
import tensorflow as tf

weight_decay = 0.0005

# mini block in resnet
def residual_block_transpose(x, shape, decrease_filter=False):
    output_filter_num = shape[1]
    if decrease_filter:
        first_stride = (2, 2)
    else:
        first_stride = (1, 1)

    pre_bn = BatchNormalization()(x)
    pre_relu = Activation('relu')(pre_bn)

    conv_1 = Conv2DTranspose(output_filter_num,
                             kernel_size=(3, 3),
                             strides=first_stride,
                             padding='SAME',
                             kernel_initializer=he_normal(),
                             kernel_regularizer=regularizers.l2(weight_decay)
                             )(pre_relu)

    bn_1 = BatchNormalization()(conv_1)
    relu1 = Activation('relu')(bn_1)
    conv_2 = Conv2DTranspose(output_filter_num,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding='SAME',
                             kernel_initializer=he_normal(),
                             kernel_regularizer=regularizers.l2(weight_decay)
                             )(relu1)

    if decrease_filter:  # change the image size and channel from x to block
        projection = Conv2DTranspose(output_filter_num,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block

def residual_block(x, shape, filter_type=''):
    output_filter_num = shape[1]
    if filter_type == 'increase':   # set the stride to (2, 2) is just like pooling
        first_stride = (2, 2)
    elif filter_type == 'decrease':
        x = UpSampling2D()(x)         # if filter is decrease, we Unsample the x first
        first_stride = (1, 1)
    else:
        first_stride = (1, 1)

    pre_bn = BatchNormalization()(x)
    pre_relu = Activation('relu')(pre_bn)

    conv_1 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=first_stride,    # if 'increase', change the feature map size here (pooling)
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(pre_relu)
    bn_1 = BatchNormalization()(conv_1)
    relu1 = Activation('relu')(bn_1)
    conv_2 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(relu1)
    if filter_type == 'increase': # change the image size and channel from x to block
        projection = Conv2D(output_filter_num,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    elif filter_type == 'decrease':
        projection = Conv2D(output_filter_num,
                            kernel_size=(1,1),
                            strides=(1, 1),
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


# abstract class for hash model
# you have to define the children class to inherate the __init__ and net_loss function,
# and name your hash_layer to 'hash_x'

class HashModel:
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.stack_num = stack_num
        self.hash_bits = hash_bits

    # you have to overite this function for the inherrent class
    def net_loss(self, y_true, y_pred):
        return 0


# supervised model for hash without use of autoencoder
class HashSupervisedModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta):
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)

        self.alpha = alpha
        self.beta = beta
        # build the supervised model here
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name='img_input')

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   )(self.img_input)

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = residual_block(x, [16, 32], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [16, 32])

        x = residual_block(x, [32, 64], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 64])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        self.hash_x = Dense(hash_bits, activation='sigmoid', kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay), name="hash_x")(x)
        self.y_supervised_output = Dense(num_classes, activation='softmax', kernel_initializer=he_normal(),
                                         kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)

    def net_loss(self, y_true, y_pred):
        supervised_loss = objectives.categorical_crossentropy(y_true, y_pred)
        binary_loss = - tf.reduce_mean(tf.square(self.hash_x - 0.5))
        balance_loss = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x, 0) - 0.5))

        return supervised_loss + self.alpha * binary_loss + self.beta * balance_loss


class HashAutoEncoderModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta):
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)
        self.alpha = alpha
        self.beta = beta
        ############################# build the auto-encoder model #####################
        ## bulid the encoder model
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name="img_input")

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   )(self.img_input)

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = residual_block(x, [16, 32], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [16, 32])

        x = residual_block(x, [32, 64], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 64])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        shape_restore = x.get_shape().as_list()[1:4]
        units_restore = shape_restore[0]*shape_restore[1]*shape_restore[2]
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        self.hash_x = Dense(hash_bits, activation='sigmoid', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay), name="hash_x")(x)

        ## build the decoder model
        x = Dense(units_restore, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)

        x = Reshape((shape_restore[0], shape_restore[1], shape_restore[2]))(x)

        for _ in range(1, self.stack_num):
            x = residual_block_transpose(x, [64, 64])
        x = residual_block_transpose(x, [64, 32], decrease_filter=True)

        for _ in range(1, self.stack_num):
            x = residual_block_transpose(x, [32, 32])
        x = residual_block_transpose(x, [32, 16], decrease_filter=True)

        for _ in range(0, self.stack_num):
            x = residual_block_transpose(x, [16, 16])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        self.y_decoded = Conv2D(filters=self.img_channels,
                                activation='sigmoid',
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay),
                                name='y_decoded'
                                )(x)
    def net_loss(self, y_true, y_pred):
        decoded_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        binary_loss = - tf.reduce_mean(tf.square(self.hash_x - 0.5))
        balance_loss = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x, 0) - 0.5))

        return decoded_loss + self.alpha * binary_loss + self.beta * balance_loss


# SIMO model
class HashSupervisedAutoEncoderModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma):
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # build the supervised autoencoder model
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name="img_input")

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   )(self.img_input)

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = residual_block(x, [16, 32], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [16, 32])

        x = residual_block(x, [32, 64], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 64])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        shape_restore = x.get_shape().as_list()[1:4]
        units_restore = shape_restore[0] * shape_restore[1] * shape_restore[2]
        x = Flatten()(x)
        self.hash_x = Dense(hash_bits, activation='sigmoid', kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay), name="hash_x")(x)

        ## build the decoder model
        x = Dense(units_restore, activation='relu', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)

        x = Reshape((shape_restore[0], shape_restore[1], shape_restore[2]))(x)

        for _ in range(1, self.stack_num):
            x = residual_block(x, [64, 64])
        x = residual_block(x, [64, 32], filter_type='decrease')

        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 32])
        x = residual_block(x, [32, 16], filter_type='decrease')

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        self.y_decoded = Conv2D(filters=self.img_channels,
                                activation='sigmoid',
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay),
                                name='y_decoded'
                                )(x)

        self.y_predict = Dense(self.num_classes, activation='softmax', kernel_initializer=he_normal(),
                               kernel_regularizer=regularizers.l2(weight_decay), name='y_predict')(self.hash_x)

    def net_loss(self, y_true, y_pred):

        supervised_loss = 0 #objectives.categorical_crossentropy(y_true, y_pred)  # we calculate supervised loss outside
        binary_loss = - tf.reduce_mean(tf.square(self.hash_x - 0.5))
        balance_loss = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x, 0) - 0.5))
        decoded_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        return supervised_loss + self.alpha * binary_loss + self.beta * balance_loss + self.gamma * decoded_loss


class HashMultiScaleAutoEncoderModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta):
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)
        self.alpha = alpha
        self.beta = beta

        # build the multi scale auto-encoder model
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name="img_input")
        x = Conv2D(16, kernel_size=(5, 5), padding='SAME', activation='relu', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(self.img_input)
        x = MaxPooling2D()(x)
        x = Conv2D(32, kernel_size=(5, 5), padding='SAME', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(64, kernel_size=(5, 5), padding='SAME', kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)

        x = Dense(256, activation='relu', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Dense(4 * 4 * 64, activation='relu', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Reshape([4, 4, 64])(x)

        hidden_shape = x.get_shape().as_list()
        hidden_size = hidden_shape[1] * hidden_shape[2] * hidden_shape[3]
        fc = Flatten()(x)
        fc = Dense(128, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(fc)
        self.hash_x = Dense(self.hash_bits, activation='sigmoid', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay), name='hash_x')(fc)
        fc = Dense(128, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)
        fc = Dense(hidden_size, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(fc)
        fc = Reshape((hidden_shape[1], hidden_shape[2], hidden_shape[3]))(fc)

        x = keras.layers.concatenate([x, fc])

        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, kernel_size=(5, 5), padding='SAME', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, kernel_size=(5, 5), padding='SAME', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(16, kernel_size=(5, 5), padding='SAME', kernel_initializer=he_normal(), kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        self.y_decoded = Conv2D(self.img_channels, kernel_size=(5, 5), activation='sigmoid', padding='SAME', kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay), name='y_decoded')(x)



    def net_loss(self, y_true, y_pred):
        print self.beta
        loss_decoded = objectives.mse(y_true, y_pred)
        # binary_loss = - tf.reduce_mean(tf.square(self.hash_x - 0.5))
        # balance_loss = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x, 0) - 0.5))

        return loss_decoded #+ self.alpha * binary_loss + self.beta * balance_loss
