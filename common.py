from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU,Dropout
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys

class ProcessImage:
    def __init__(self):
        pass

    def load_image(self):
        dom_A = glob.glob("image_domainA/*")
        dom_B = glob.glob("image_domainB/*")
        image_A = [cv2.imread(i) for i in dom_A]
        image_B = [cv2.imread(i) for i in dom_B]
        image_A = image_B[0:len(image_B)]
        image_A = [cv2.resize(i,(216, 216))/255 for i in image_A]
        image_B = [cv2.resize(i,(216,216))/255 for i in image_B]
        return np.array([image_A],dtype=np.float32), np.array([image_B],dtype=np.float32)

class ConvBlock(Model):
    def __init__(self, filter_num: int, kernel_size: int = 3, stride: int = 1, padding_num: int = 0, output_padding: int = 1, do_padding=False, do_frac=False, do_leakyr=False, do_norm=True, do_relu=True, padding='same', *args, **kwargs):  
        super(ConvBlock, self).__init__(*args, **kwargs)
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.do_frac = do_frac
        self.padding_num = padding_num
        self.do_leakyr = do_leakyr
        self.do_padding = do_padding
        self.do_norm = do_norm
        self.do_relu = do_relu
        if self.do_frac:
            self.conv = Conv2DTranspose(filters=self.filter_num, kernel_size=self.kernel_size, strides=int(self.stride ** (-1)),padding='same',output_padding=output_padding)
        else:
            self.conv = Conv2D(filters=self.filter_num, kernel_size=self.kernel_size, strides=self.stride, padding=padding)
        self.norm=tfa.layers.InstanceNormalization(axis=3, 
                                               center=True, 
                                               scale=True,
                                               beta_initializer="random_uniform",
                                               gamma_initializer="random_uniform")
        if self.do_leakyr:
            self.relu = LeakyReLU()
        else:
            self.relu = ReLU()
        
    
    def call(self, inputs):
        if self.do_padding:
            inputs = tf.pad(inputs, [[self.padding_num,self.padding_num] for _ in range(len(inputs.shape))], "REFLECT")
        tensor = self.conv(inputs)
        if self.do_norm:
            tensor = self.norm(tensor)
        return self.relu(tensor) if self.do_relu else tensor


class ResNN(Model):
    def __init__(self, unit_num=9, *args, **kwargs):
        super(ResNN, self).__init__(*args, **kwargs)
        self.unit_num = unit_num
        self.conv_res = Conv2D(filters=256, kernel_size=(3, 3),padding='same')
        self.resList_1: List[Layers] = []
        self.resList_2: List[Layers] = []
    
    def call(self, inputs):
        for i in range(self.unit_num // 2):
            self.resList_1.append(self.conv_res(inputs))
            inputs = self.resList_1[-1]
        inputs = self.conv_res(inputs)
        for i in range(self.unit_num // 2):
            inputs += self.resList_1[-(i + 1)]
            self.resList_2.append(self.conv_res(inputs))
            inputs = self.resList_2[-1]
        return inputs