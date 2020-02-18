from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU, Dropout, AveragePooling2D, LayerNormalization,Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model,Sequential
#import tensorflow_addons as tfa
import numpy as np
import random
import sys
from common import ProcessImage, ConvBlock, ResNN
batch_size = 1

class Generator_block_1(Model):
    def __init__(self, filter_num, dropout_rate: float = 0.0, *args, **kwargs):
        super(Generator_block_1, self).__init__(*args, **kwargs)
        self.conv_1 = ConvBlock(filter_num=filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True, do_relu=False,padding='valid')
        self.conv_3 = ConvBlock(filter_num=filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True, do_relu=False,padding='valid')
        self.conv_2 = ConvBlock(filter_num=filter_num*2, kernel_size=1, stride=1, padding_num=0, do_padding=False, do_norm=False,padding='valid')
        self.conv_2_1 = ConvBlock(filter_num=filter_num, kernel_size=1, stride=1, padding_num=0, do_padding=False, do_norm=False,padding='valid')
        self.conv_4 = ConvBlock(filter_num=filter_num*2, kernel_size=1, stride=1, padding_num=0, do_padding=False, do_norm=False,padding='valid')
        self.conv_4_1 = ConvBlock(filter_num=filter_num, kernel_size=1, stride=1, padding_num=0, do_padding=False, do_norm=False,padding='valid')
        self.dropout = Dropout(rate=dropout_rate)
    
    def call(self, input_, z):
        z = tf.reshape(z, [-1, 1, 1, z.shape[3]])
        z_expand = tf.tile(z, [-1, input_.shape[1], input_.shape[2], 1])
        out = self.conv_1(input_)
        out = self.conv_2(tf.concat([out, z_expand], -1))
        out = self.conv_2_1(out)
        out = self.conv_3(out)
        out = self.conv_4(tf.concat([out, z_expand], -1))
        out = self.conv_4_1(out)
        return self.dropout(out) + input_


class Generator_block_2(Model):
    def __init__(self, filter_num, *args, **kwargs):
        super(Generator_block_2, self).__init__(*args, **kwargs)
        self.conv = Conv2DTranspose(filters=filter_num, kernel_size=3, strides=2, padding='same')
        self.norm = LayerNormalization()
        self.relu = ReLU()
    def call(self, input_):
        return self.relu(self.norm(self.conv(input_)))

class Generator_block_3(Model):
    def __init__(self, *args, **kwargs):
        super(Generator_block_3, self).__init__(*args, **kwargs)
        self.dense_1 = Dense(256, activation='relu')
        self.dense_2 = Dense(256, activation='relu')
        self.dense_3 = Dense(256 * 4)
    
    def call(self, input_):
        return self.dense_3(self.dense_2(self.dense_1(input_)))        

class Generator(Model):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        filter_num = 256
        self.block_1 = [Generator_block_1(filter_num=filter_num) for i in range(4)]
        self.block_2 = []
        for _ in range(2):
            filter_num = int(filter_num / 2)
            self.block_2 += [Generator_block_2(filter_num=filter_num)]
        self.conv = Conv2DTranspose(filters=3, kernel_size=1, strides=1, padding='valid')
        self.tanh = Activation(activation=tf.nn.tanh)
        self.block_3 = Generator_block_3()

    def call(self, z, input_):
        z_out = self.block_3(z)
        #各ブロック用に分割を行う [x,y,z,h] -> [x,y,z,h/4] x 4 (h==0(mod4))
        z1, z2, z3, z4 = tf.split(z_out, [256 for _ in range(4)], -1)
        z_list = [z1, z2, z3, z4]
        for i, (z, layer) in enumerate(zip(z_list, self.block_1)):
            out = layer(out, z) if i != 0 else layer(input_, z)
        for layer in self.block_2:
            out = layer(out)
        return self.tanh(self.conv(out))

class Discriminator_content(Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator_content, self).__init__(*args, **kwargs)
        self.convs = [ConvBlock(256, kernel_size=7, stride=2, padding_num=1, do_padding=True, do_leakyr=True) for _ in range(3)]
        self.convs += [ConvBlock(256, kernel_size=4, stride=1, do_leakyr=True)]
        self.flat = Flatten()
        self.dense = Dense(1, activation='sigmoid')
    
    def call(self, input_):
        for i, layer in enumerate(self.convs):
            out = layer(out) if i != 0 else layer(input_)
        return self.dense(self.flat(out))

class Discriminator(Model):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        filter_num = 64
        self.convs = [ConvBlock(filter_num=filter_num if i == 0 else filter_num * (2 ** i), kernel_size=3, stride=2, padding_num=1, do_padding=True if i != 5 else False) for i in range(6)]
        self.flat = Flatten()
        self.dense = Dense(1, activation='sigmoid')
    
    def call(self, input_):
        for i, layer in enumerate(self.convs):
             out = layer(out) if i != 0 else layer(input_)
        return self.dense(self.flat(out))


class Encoder_Block_c(Model):
    def __init__(self, dropout_rate: float = 0.0, *args, **kwargs):
        super(Encoder_Block_c, self).__init__(*args, **kwargs)
        self.filter_num = 64
        self.dropout_rate = dropout_rate
        self.convs_1 = []
        self.convs_2 = []

        self.dropout = Dropout(rate=self.dropout_rate)
        #LeakyReLUConv2d
        #zeropaddingを3個つけるのはtensorflowのcon2dでは無理っぽい
        self.conv_1 = ConvBlock(self.filter_num, kernel_size=7, stride=1, padding_num=3, do_leakyr=True, do_padding=True,padding='valid')

        #ReLuINSConv2d
        for _ in range(2):
            self.convs_1 += [ConvBlock(self.filter_num * 2, kernel_size=3, stride=2, padding_num=1, do_padding=True,padding='valid')]
            self.filter_num *= 2
    
        #INSResBlock
        for _ in range(3):
            self.convs_2 += [
            [ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True,padding='valid'),
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True, do_relu=False,padding='valid'),
            self.dropout]
            ]
        
    def call(self, inputs):
        output = self.conv_1(inputs)
        for layer in self.convs_1:
            output = layer(output)
        for layers in self.convs_2:
            for i,layer in enumerate(layers):
                output_tmp = layer(output) if i==0 else layer(output_tmp)
            output += output_tmp
        return output
        

class Encoder_content(Model):
    def __init__(self, dropout_rate: int = 0.0, *args, **kwargs):
        super(Encoder_content, self).__init__(*args, **kwargs)
        self.dropout_rate = dropout_rate
        self.convs_common = []
        self.encoder_A = Encoder_Block_c(dropout_rate=self.dropout_rate)
        self.encoder_B = Encoder_Block_c(dropout_rate=self.dropout_rate)
        self.filter_num = self.encoder_A.filter_num
        self.dropout = Dropout(rate=self.dropout_rate)
        for _ in range(1):
            self.convs_common += [
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True,padding='valid'),
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True, do_relu=False,padding='valid'),
            self.dropout
            ]
    def call(self, input_A, input_B):
        def forward_common(layers, input_data):
            for layer in layers:
                input_data = layer(input_data)
            return input_data
        
        #self.input_A = Input(shape=(img_dim, img_dim, img_channel))
        #self.input_B = Input(shape=(img_dim, img_dim, img_channel))
        output_A = self.encoder_A(input_A)
        output_B = self.encoder_B(input_B)
        self.output_A = forward_common(self.convs_common, output_A)
        self.output_B = forward_common(self.convs_common, output_B)
        return self.output_A, self.output_B
        #self.model = Model(inputs=[self.input_A, self.input_B], outputs=[self.output_A, self.output_B])
        #self.model.summary()
        

class Encoder_attribute(Model):
    def __init__(self, *args, **kwargs):
        super(Encoder_attribute, self).__init__(*args, **kwargs)
        filters_num = [64, 128, 256, 256, 256]
        #self.input = Input(shape=(img_dim, img_dim, img_channel))
        self.convs = [ConvBlock(filter_num=filters_num[0], kernel_size=7, stride=1, padding_num=3, do_padding=True, do_norm=False, do_relu=False)]
        for i in range(4):
            self.convs += [ConvBlock(filter_num=filters_num[i + 1], kernel_size=4, stride=2, padding_num=1, do_padding=True, do_norm=False)]
        
        self.conv_final = Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')
        self.pool = AveragePooling2D((16, 16))
        #self.inputs = self.input

    def call(self, inputs):
        for i, layer in enumerate(self.convs):
            inputs = layer(inputs)
        output = self.pool(inputs)
        return self.conv_final(output)
        #self.model = Model(inputs=[self.input], outputs=[self.output])
        #self.model.summary()

class ForSeeking(Model):
    def __init__(self,*args,**kwargs):
        super(ForSeeking, self).__init__(*args, **kwargs)
    
    def call(self, inputs):
        loss_1 = tfk.losses.mae(inputs[0], inputs[1])
        loss_1 = tf.reduce_mean(loss_1, axis=[1, 2])
        loss_2 = tfk.losses.mae(inputs[2], inputs[3])
        loss_2 = tf.reduce_mean(loss_2, axis=[1, 2])
        return loss_2 / loss_1



class Lossfunc:

    def content_adversarial_eg(self, y_true, y_pred):
        return - tfk.losses.binary_crossentropy(y_true, y_pred)
    
    def domain_adversarial_eg(self, y_true, y_pred):
        return -tfk.losses.binary_crossentropy(y_true, y_pred)
    
    def content_adversarial(self, y_true, y_pred):
        return tfk.losses.binary_crossentropy(y_true, y_pred)
    
    def domain_adversarial(self, y_true, y_pred):
        return tfk.losses.binary_crossentropy(y_true, y_pred)

    def cross_cycle_consistensy(self, y_true, y_pred):
        loss_2 = tfk.losses.mae(y_true, y_pred)
        loss_2 = tf.reduce_mean(loss_2, axis=[1, 2])
        return loss_2
    
    def self_reconstruction(self, y_true, y_pred):
        loss_2 = tfk.losses.mae(y_true, y_pred)
        loss_2 = tf.reduce_mean(loss_2, axis=[1, 2])
        return loss_2
    
    def mode_seeking(self, y_true, y_pred): 
        return y_pred
        

class DRIT_pp:
    def __init__(self, img_dim: int, img_channel: int):
        self.encoder_content = Encoder_content()
        self.encoder_attr_A = Encoder_attribute()
        self.encoder_attr_B = Encoder_attribute()
        self.genarator_A = Generator()
        self.genarator_B = Generator()
        self.discriminator_con_A = Discriminator_content()
        self.discriminator_con_B = Discriminator_content()
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()
        self.forseek = ForSeeking()
        self.input_A = Input(shape=(img_dim, img_dim, img_channel))
        self.input_B = Input(shape=(img_dim, img_dim, img_channel))

    def compile_model(self):
        self.discriminator_A.trainable, self.discriminator_B.trainable, self.discriminator_con_A.trainable, \
        self.discriminator_con_B.trainable = False, False, False, False
        self.e_c_A, self.e_c_B = self.encoder_content(self.input_A, self.input_B)
        self.disA = self.discriminator_con_A(self.e_c_A)
        self.disB = self.discriminator_con_B(self.e_c_B)
        self.e_attr_A = self.encoder_attr_A(self.input_A)
        self.e_attr_B = self.encoder_attr_B(self.input_B)
        self.gen_A_selfconst = self.genarator_A(self.e_attr_A, self.e_c_A)
        self.gen_B_selfconst = self.genarator_B(self.e_attr_B, self.e_c_B)
        self.genA = self.genarator_A(self.e_attr_A, self.e_c_B)
        self.genB = self.genarator_B(self.e_attr_B, self.e_c_A)
        self.e_attr_A_1 = self.encoder_attr_A(self.genA)
        self.e_attr_B_1 = self.encoder_attr_B(self.genB)
        self.e_c_A_1, self.e_c_B_1 = self.encoder_content(self.genA, self.genB)
        self.genA_1 = self.genarator_A(self.e_attr_A_1, self.e_c_B_1)
        self.genB_1 = self.genarator_B(self.e_attr_B_1, self.e_c_A_1)
        self.disA_1 = self.discriminator_A(self.genA_1)
        self.disB_1 = self.discriminator_B(self.genB_1)
        self.modeseek_1 = self.forseek([self.e_attr_A_1, self.e_attr_B_1, self.gen_A_selfconst, self.genB])
        self.modeseek_2 = self.forseek([self.e_attr_A_1, self.e_attr_B_1, self.genA, self.gen_B_selfconst])
        model_EG = Model(inputs=[self.input_A, self.input_B],
        outputs=[
        self.disA, self.disB, self.e_c_A, self.e_c_B,
        self.modeseek_1,
        self.modeseek_2,
        self.gen_A_selfconst,self.gen_B_selfconst,
        self.genA_1, self.genB_1,
        self.disA_1, self.disB_1
        ])
        model_EG.summary()
        lossfunc = Lossfunc()
        model_EG.compile(
            optimizer=tfk.optimizers.Adam(lr=0.0002),
            loss={
            'discriminator_content': lossfunc.content_adversarial_eg,
            'discriminator_content_1':lossfunc.content_adversarial_eg,
            'for_seeking':lossfunc.mode_seeking,
            'for_seeking_1': lossfunc.mode_seeking,
            'generator':lossfunc.self_reconstruction,
            'generator_1':lossfunc.self_reconstruction,
            'generator_2': lossfunc.cross_cycle_consistensy,
            'generator_1_1':lossfunc.cross_cycle_consistensy,
            'discriminator': lossfunc.domain_adversarial_eg,
            'discriminator_1': lossfunc.domain_adversarial_eg
            }
        )
        model_D_c_A = Sequential(
            Discriminator_content()
        )
        model_D_c_B = Sequential(
            Discriminator_content()
        )
        model_D_a_A = Sequential(
            Discriminator()
        )
        model_D_a_B = Sequential(
            Discriminator()
        )
        model_D_c_A.compile(
            optimizer=tfk.optimizers.Adam(lr=0.0002),
            loss={
            'discriminator_content': lossfunc.content_adversarial
            }
        )
        model_D_c_B.compile(
            optimizer=tfk.optimizers.Adam(lr=0.002),
            loss={
                'discriminator_content':lossfunc.content_adversarial
            }
        )
        model_D_a_A.compile(
            optimizer=tfk.optimizers.Adam(lr=0.002),
            loss={
                'discriminator':lossfunc.domain_adversarial
            }
        )
        model_D_a_B.compile(
            optimizer=tfk.optimizers.Adam(lr=0.002),
            loss={
                'discriminator':lossfunc.domain_adversarial
            }
        )
            
        return model_EG, model_D_c_A, model_D_c_B, model_D_a_A, model_D_a_B
