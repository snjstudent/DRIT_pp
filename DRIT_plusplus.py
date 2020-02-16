from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU, Dropout, AveragePooling2D
from tensorflow.keras.models import Model,Sequential
#import tensorflow_addons as tfa
import numpy as np
import random
import sys
from common import ProcessImage, ConvBlock, ResNN

        
    
class Generator:
    def __init__(self, img_dim, img_channel):
        self.img_dim = img_dim
        self.img_channel = img_channel
        conv_block_1 = ConvBlock(64, 7)
        conv_block_2 = ConvBlock(128, stride=2)
        conv_block_3 = ConvBlock(256, stride=2)
        conv_block_4 = ConvBlock(128, stride=1 / 2, do_frac=True)
        conv_block_5 = ConvBlock(64, stride=1 / 2, do_frac=True)
        conv_block_6 = ConvBlock(3, 7)
        resnet = ResNN(unit_num=9)

        self.input_g = Input(shape=(img_dim, img_dim, img_channel), name="input_generator")
        conv_1 = conv_block_1(self.input_g)
        conv_2 = conv_block_2(conv_1)
        conv_3 = conv_block_3(conv_2)
        res_1 = resnet(conv_3)
        conv_4 = conv_block_4(res_1)
        conv_5 = conv_block_5(conv_4)
        self.conv_6 = conv_block_6(conv_5)
        self.model = Model(inputs=[self.input_g], outputs=[self.conv_6])
        #self.model.summary()
        
class Discriminator:
    def __init__(self, img_dim, img_channel):
        self.img_dim = img_dim
        self.img_channel = img_channel
        list_conv: List[Model] = []
        input_d = Input(shape=(img_dim, img_dim, img_channel))
        conv_1 = ConvBlock(64, 4,do_leakyr=True)
        list_conv.append(conv_1(input_d))
        for i in range(3):
            conv_tmp = ConvBlock(64 * (2 ** (i + 1)), 4, do_leakyr=True)
            list_conv.append(conv_tmp(list_conv[-1]))
        flat = Flatten()(list_conv[-1])
        self.out = Dense(1, activation='sigmoid')(flat)
        self.model = Model(inputs=[input_d], outputs=[self.out])
        #self.model.summary()

class Encoder_Block_c(Model):
    def __init__(self, dropout_rate: int = 0.0, *args, **kwargs):
        super(Encoder_Block_c, self).__init__(*args, **kwargs)
        self.filter_num = 64
        self.dropout_rate = dropout_rate
        self.convs_1 = []
        self.convs_2 = []

        self.dropout = Dropout(rate=self.dropout_rate)
        #LeakyReLUConv2d
        #zeropaddingを3個つけるのはtensorflowのcon2dでは無理っぽい
        self.conv_1 = ConvBlock(self.filter_num, kernel_size=7, stride=3, padding_num=0, do_leakyr=True, do_padding=True)

        #ReLuINSConv2d
        for _ in range(2):
            self.convs_1 += [ConvBlock(self.filter_num * 2, kernel_size=3, stride=2, padding_num=1, do_padding=True)]
            self.filter_num *= 2
    
        #INSResBlock
        for _ in range(3):
            self.convs_2 += [
            [ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True),
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True),
            self.dropout]
            ]
        
    def call(self, inputs):
        output = self.conv_1(inputs)
        for layer in self.convs_1:
            output = layer(output)
        for layers in self.convs_2:
            for layer in layers:
                output_tmp = layer(output)
            output += output_tmp
        return output
        

class Encoder_content:
    def __init__(self, img_dim, img_channel, dropout_rate: int = 0.0):
        self.dropout_rate = dropout_rate
        self.convs_common = []
        self.encoder_A = Encoder_Block_c(dropout_rate=self.dropout_rate)
        self.encoder_B = Encoder_Block_c(dropout_rate=self.dropout_rate)
        self.filter_num = self.encoder_A.filter_num
        self.dropout = Dropout(rate=self.dropout_rate)
        for _ in range(3):
            self.convs_common += [
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True),
            ConvBlock(self.filter_num, kernel_size=3, stride=1, padding_num=1, do_padding=True),
            self.dropout
            ]
        def forward_common(layers, input_data):
            for layer in layers:
                input_data = layer(input_data)
            return input_data
        
        self.input_A = Input(shape=(img_dim, img_dim, img_channel))
        self.input_B = Input(shape=(img_dim, img_dim, img_channel))
        output_A = self.encoder_A(self.input_A)
        output_B = self.encoder_B(self.input_B)
        self.output_A = forward_common(self.convs_common, output_A)
        self.output_B = forward_common(self.convs_common, output_B)
        self.model = Model(inputs=[self.input_A, self.input_B], outputs=[self.output_A, self.output_B])
        self.model.summary()
        

class Encoder_attribute:
    def __init__(self, img_dim, img_channel):
        filters_num = [64, 128, 256, 256, 256]
        self.input = Input(shape=(img_dim, img_dim, img_channel))
        convs = [ConvBlock(filter_num=filters_num[0], kernel_size=7, stride=1, padding_num=3, do_padding=True, do_norm=False, do_relu=False)]
        for i in range(4):
            convs += [ConvBlock(filter_num=filters_num[i + 1], kernel_size=4, stride=2, padding_num=1, do_padding=True, do_norm=False)]
        
        self.conv_final = Conv2D(filters=8, kernel_size=1, strides=1, padding='valid')
        self.inputs = self.input
        for layer in convs:
            self.inputs = layer(self.inputs)
        self.pool = AveragePooling2D((self.inputs.shape[1], self.inputs.shape[2]))
        output = self.pool(self.inputs)
        self.output = self.conv_final(output)
        self.model = Model(inputs=[self.input], outputs=[self.output])
        self.model.summary()


class Lossfunc:
    def __init__(self, lamda):
        self.lamda = 10
        self.count = 1
        self.batch_size = 1

    def get_target_model(self, model_target):
        self.model_target = model_target
    
    def get_self_model(self, self_model):
        self.self_model = self_model
        
    def set_input_data(self, input_data):
        self.input_data = input_data
        self.pred_datas = input_data
    
    def set_predict_data(self, input_data):
        self.pred_data = input_data
    
    def set_batch_size(self, batch_size):
        self.count = 1
        self.batch_size = batch_size

    def predict_data(self):
        self.pred_datas = self.model_target.predict(self.self_model.predict(self.input_data))
    
    def l1_norm_loss(self, y_true, y_pred):
        loss_1 = tfk.losses.binary_crossentropy(y_true, y_pred)
        loss_2 = tfk.losses.mae(self.input_data[((self.count - 1) * self.batch_size):(self.count * self.batch_size)], self.pred_datas[((self.count - 1) * self.batch_size):(self.count * self.batch_size)])
        loss_2 = tf.reduce_mean(loss_2, axis=[1,2])
        self.count += 1
        if (self.count * self.batch_size >= len(self.input_data)):
            self.count = 1
        loss_l1 = (loss_2 * self.lamda) - loss_1
        return loss_l1

class CycleGAN:
    def __init__(self):
        self.Gen_G = Generator(256, 3)
        self.Gen_F = Generator(256, 3)
        self.Dis_G = Discriminator(256, 3)
        self.Dis_F = Discriminator(256, 3)
        self.Dis_G_1 = Discriminator(256, 3)
        self.Dis_F_1 = Discriminator(256, 3)
        self.Loss_cycle_1 = Lossfunc(10)
        self.Loss_cycle_2 = Lossfunc(10)
        
        self.model_1 = Sequential([
            self.Gen_G.model,
            self.Dis_G_1.model
        ])
        self.model_2 = Sequential([
            self.Gen_F.model,
            self.Dis_F_1.model
        ])
    def compile_model(self):
        #self.mode_tm = Model(inputs=[Gen_G.input_g, Gen_G.conv_6], outputs=[Gen_G.conv_6,Dis_G.out])
        self.Gen_G.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Gen_F.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Loss_cycle_1.get_target_model(self.Gen_F.model)
        self.Loss_cycle_1.get_self_model(self.Gen_G.model)
        self.Loss_cycle_2.get_target_model(self.Gen_G.model)
        self.Loss_cycle_2.get_self_model(self.Gen_F.model)
        self.model_1.layers[1].trainable = False
        self.model_2.layers[1].trainable = False
        self.model_1.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss=self.Loss_cycle_1.l1_norm_loss)
        self.model_2.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss=self.Loss_cycle_2.l1_norm_loss)
        self.Dis_G.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Dis_F.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
    
    def make_randomdata(self, input_, model_self, model_target):
        pred_datas = model_target.predict(model_target.predict(input_))
        datas = []
        label = []
        for i in range(len(input_)):
            r = random.uniform(0, 1)
            if r > 0.5:
                datas.append(pred_datas[i])
                label.append(0)
            else:
                datas.append(input_[i])
                label.append(1)
        return np.array(label, dtype=np.float32), np.array(datas, dtype=np.float32)
    """           
    def save_model(self,i):
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        upload_file = drive.CreateFile()
        self.Dis_G.model.save_weights("Dis_G_"+str(i)+".hdf5")
        self.Dis_F.model.save_weights("Dis_F_"+str(i)+".hdf5")
        self.model_1.save_weights("model_1_"+str(i)+".hdf5")
        self.model_2.save_weights("model_2_"+str(i)+".hdf5")
        upload_file.SetContentFile("Dis_G_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("Dis_F_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("model_1_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("model_2_"+str(i)+".hdf5")
        upload_file.Upload()
    """

    def train(self, input_A, input_B,count, batch_size=1, steps=1):
        self.Loss_cycle_1.set_input_data(input_A)
        self.Loss_cycle_2.set_input_data(input_B)
        self.Loss_cycle_1.set_batch_size(batch_size)
        self.Loss_cycle_2.set_batch_size(batch_size)
        b = np.array([0] * len(input_B), dtype=np.float32)
        self.Loss_cycle_1.batch_size = batch_size
        self.Loss_cycle_2.batch_size = batch_size
        for i in range(steps):
            #print("Step : ", i + 1)
            label_disG, datas_DisG = self.make_randomdata(input_A, self.Gen_G.model, self.Gen_F.model)
            if count%3==0:
              label_disF, datas_DisF = self.make_randomdata(input_B, self.Gen_F.model, self.Gen_G.model)
              self.Dis_G.model.fit(datas_DisG, label_disG, batch_size=batch_size)
              self.Dis_F.model.fit(datas_DisF, label_disF, batch_size=batch_size)
            self.model_1.layers[1].set_weights(self.Dis_G.model.get_weights())
            self.model_2.layers[1].set_weights(self.Dis_F.model.get_weights())
            self.Loss_cycle_1.predict_data()
            self.Loss_cycle_2.predict_data()
            self.model_1.fit(input_A, b, batch_size=batch_size)
            self.model_2.fit(input_B, b, batch_size=batch_size)
            testimage = self.Gen_F.model.predict(datas_DisG, batch_size=batch_size)
            cv2.imwrite("test.png", np.array(testimage[0]) * 255.0)
            print("")
            print("")
            print("image created")
            print("")
            print("")      


if __name__ == "__main__":
    Encoder_content(216,3)
    Encoder_attribute(216,3)

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    processor = ProcessImage()
    A, B = processor.load_image()
    #tf.executing_eagerly()
    #tf.enable_eager_execution()
    cycleGAN = CycleGAN()
    cycleGAN.Loss_cycle_1.set_input_data(A[0][0:100])
    cycleGAN.Loss_cycle_2.set_input_data(B[0][0:100])
    #cycleGAN.Loss_cycle_1.set_predict_data(A[0][0])
    #cycleGAN.Loss_cycle_2.set_predict_data(B[0][0])
    cycleGAN.compile_model()
    for i in range(1,10000):
        
        if (i==1 or i%5==0):
          cycleGAN.save_model(i)

        print("Step : ", i + 1)
        A_batch = np.random.permutation(A[0])
        B_batch = np.random.permutation(B[0])
        for u in range(int(len(A_batch) / 100)):
            cycleGAN.Loss_cycle_1.set_input_data(A_batch[u * 100:(u + 1) * 100])
            cycleGAN.Loss_cycle_2.set_input_data(B_batch[u * 100:(u + 1) * 100])
            cycleGAN.train(A_batch[u * 100:(u + 1) * 100], B_batch[u * 100:(u + 1) * 100],u, batch_size=4, steps=1)
        """