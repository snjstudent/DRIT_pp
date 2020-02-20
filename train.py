from __future__ import absolute_import, division, print_function
from DRIT_plusplus import DRIT_pp, batch_size
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


class Train:
    def make_randomdata(self, input_, model):
        pred_datas = model.predict(input_, batch_size=batch_size)
        preds_cA, preds_cB, preds_imgA, preds_imgB = pred_datas[2], pred_datas[3], pred_datas[8], pred_datas[9]
        datas_cA, datas_cB, imgs_A, imgs_B = [], [], [], []
        labels = []
        for cA, cB, img_A, img_B in zip(preds_cA, preds_cB, preds_imgA, preds_imgB):
            r = random.uniform(0, 1)
            if r > 0.5:
                datas_cA.append(cB)
                datas_cB.append(cA)
                imgs_A.append(img_B)
                imgs_B.append(img_A)
                labels.append(0)
            else:
                datas_cA.append(cA)
                datas_cB.append(cB)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
                labels.append(1)
        labels_cA, labels_cB, labels_imgA, labels_imgB = labels, labels, labels, labels
        return np.array(datas_cA, dtype=np.float32), np.array(datas_cB, dtype=np.float32), np.array(imgs_A, dtype=np.float32), np.array(imgs_B, dtype=np.float32), np.array(labels, dtype=np.float32)
      
    def set_weights(self, model, model_dca, model_dcb, model_daa, model_dab):
        model.layers[7].set_weights(model_dca.get_weights())
        model.layers[8].set_weights(model_dcb.get_weights())
        model.layers[10].set_weights(model_daa.get_weights())
        model.layers[11].set_weights(model_dab.get_weights())
    
    def save_weights(self, model, model_dca, model_dcb, model_daa, model_dab):
        model.save_weights("model" + str(i) + ".hdf5")
        model_dca.save_weights("model_dca" + str(i) + ".hdf5")
        model_dcb.save_weights("model_dcb" + str(i) + ".hdf5")
        model_daa.save_weights("model_daa" + str(i) + ".hdf5")
        model_dab.save_weights("model_dab" + str(i) + ".hdf5")

    def load_weights(self, model, model_dca, model_dcb, model_daa, model_dab, num):
        if num != 0:
            model.load_weights("model" + str(num) + ".hdf5")
            model_dca.load_weights("model_dca" + str(num) + ".hdf5")
            model_dcb.load_weights("model_dcb" + str(num) + ".hdf5")
            model_daa.load_weights("model_daa" + str(num) + ".hdf5")
            model_dab.load_weights("model_dba" + str(num) + ".hdf5")
        else : return
        
    def display_image(self, model, input_):
        preds = model.predict(input_, batch_size=batch_size)
        pred_image = preds[8]
        cv2.imwrite("test.png", np.array(pred_image[0] * 255.0))
        
    def train(self, model, model_dca, model_dcb, model_daa, model_dab, input_A, input_B, steps: int = 1, batch_size: int = 1):
        self.load_weights(model, model_dca, model_dcb, model_daa, model_dab, 0)
        count = int(min(len(input_A), len(input_B)) / 100)
        model_label = np.array([0] * 100, dtype=np.float32)
        encoder_label = np.array([[[[0 for _ in range(256)] for _ in range(54)] for _ in range(54)] for _ in range(100)], dtype=np.float32)
        for i in range(steps):
            print("Step : " + str(i + 1))
            A = np.random.permutation(input_A)
            B = np.random.permutation(input_B)
            batch_dataA, batch_dataB = A[0:100], B[0:100]
            for u in range(count):
                batch_dataA, batch_dataB = A[u * 100:(u + 1) * 100], B[u * 100:(u + 1) * 100]
                if i != 0 or u != 0:
                    self.set_weights(model, model_dca, model_dcb, model_daa, model_dab)
                cA, cB, imgA, imgB, label = self.make_randomdata([batch_dataA, batch_dataB], model)
                model_daa.fit(imgA, label, batch_size=batch_size)
                model_dab.fit(imgB, label, batch_size=batch_size)
                model_dca.fit(cA, label, batch_size=batch_size)
                model_dcb.fit(cB, label, batch_size=batch_size)
                model.fit([batch_dataA, batch_dataB],
                {'discriminator_content': model_label, 'discriminator_content_1': model_label,
                'encoder_content':encoder_label,'encoder_content_1':encoder_label,
                'for_seeking': model_label, 'for_seeking_1': model_label,
                'generator': batch_dataA, 'generator_1': batch_dataB,
                'generator_2': batch_dataA, 'generator_1_1': batch_dataB,
                'discriminator': model_label, 'discriminator_1': model_label}
                ,batch_size=batch_size)
            self.save_weights()
            self.display_image(model, [batch_dataA, batch_dataB])
            self.save_weights(model, model_dca, model_dcb, model_daa)

                
if __name__ == "__main__":
    processor = ProcessImage()
    A, B = processor.load_image()
    tf.executing_eagerly()
    #tf.contrib.eager.enable_eager_execution()
    trainer = Train()
    drit = DRIT_pp(216, 3)
    model, model_dca, model_dcb, model_daa, model_dab = drit.compile_model()
    trainer.train(model, model_dca, model_dcb, model_daa, model_dab, A, B, steps=10000, batch_size=batch_size)
    
    
                    

        
        
    
