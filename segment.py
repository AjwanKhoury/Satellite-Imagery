import os
import numpy as np 

import cv2 as cv
import matplotlib.pyplot as plt

from tqdm import tqdm

import keras 
from keras import Sequential 
from keras.metrics import MeanIoU
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Input, Conv2DTranspose, concatenate, GlobalAveragePooling2D, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import warnings
warnings.filterwarnings('ignore')

# Contraction 
class EncoderBlock(keras.layers.Layer):    
    def __init__(self, filters, rate=None, pooling=True):
        super(EncoderBlock,self).__init__()
        self.filters = filters
        self.rate = rate
        self.pooling = pooling
        self.conv1 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.filters,kernel_size=3,strides=1,padding='same',activation='relu',kernel_initializer='he_normal')
        if self.pooling: self.pool = MaxPool2D(pool_size=(2,2))
        if self.rate is not None: self.drop = Dropout(rate)
        
    def call(self, inputs):
        x = self.conv1(inputs)
        if self.rate is not None: x = self.drop(x)
        x = self.conv2(x)
        if self.pooling: 
            y = self.pool(x)
            return y, x
        else:
            return x
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            "filters":self.filters,
            "rate":self.rate,
            "pooling":self.pooling
        }

# Expansion
class DecoderBlock(keras.layers.Layer):

    def __init__(self, filters, rate=None, axis=-1):
        super(DecoderBlock,self).__init__()
        self.filters = filters
        self.rate = rate
        self.axis = axis
        self.convT = Conv2DTranspose(self.filters,kernel_size=3,strides=2,padding='same')
        self.conv1 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')
        if rate is not None: self.drop = Dropout(self.rate)
        self.conv2 = Conv2D(self.filters, kernel_size=3, activation='relu', kernel_initializer='he_normal', padding='same')
        
    def call(self, inputs):
        X, short_X = inputs
        ct = self.convT(X)
        c_ = concatenate([ct, short_X], axis=self.axis)
        x = self.conv1(c_)
        if self.rate is not None: x = self.drop(x)
        y = self.conv2(x)
        return y
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config, 
            "filters":self.filters,
            "rate":self.rate,
            "axis":self.axis,
        }

# Post Process
def post_process(image,threshold=0.4):
    return image>threshold
    
def init_model():
    SIZE = 128
    # build adaptive threshold UNET model
    inputs= Input(shape=(SIZE,SIZE,3))

    # Contraction 
    p1, c1 = EncoderBlock(16,0.1)(inputs)
    p2, c2 = EncoderBlock(32,0.1)(p1)
    p3, c3 = EncoderBlock(64,0.2)(p2)
    p4, c4 = EncoderBlock(128,0.2)(p3)

    # Encoding Layer
    c5 = EncoderBlock(256,rate=0.3,pooling=False)(p4) 

    # Expansion
    d1 = DecoderBlock(128,0.2)([c5,c4]) # [current_input, skip_connection]
    d2 = DecoderBlock(64,0.2)([d1,c3])
    d3 = DecoderBlock(32,0.1)([d2,c2])
    d4 = DecoderBlock(16,0.1, axis=3)([d3,c1])

    # Outputs
    outputs = Conv2D(1,1,activation='sigmoid')(d4)

    unet = keras.models.Model(
        inputs=[inputs],
        outputs=[outputs])
    return unet

def build_model():
    images = []
    mask = []

    image_path = 'data/Segment/Images/'
    mask_path = 'data/Segment/Masks/'

    image_names = sorted(next(os.walk(image_path))[-1])
    mask_names = sorted(next(os.walk(mask_path))[-1])

    # create image and mask lists
    SIZE = 128
    images = np.zeros(shape=(len(image_names),SIZE, SIZE, 3))
    masks = np.zeros(shape=(len(image_names),SIZE, SIZE, 1))

    for id in tqdm(range(len(image_names)), desc="Images"):
        path = image_path + image_names[id]
        img = img_to_array(load_img(path)).astype('float')/255.
        img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
        images[id] = img

    for id in tqdm(range(len(mask_names)), desc="Mask"):
        path = mask_path + mask_names[id]
        mask = img_to_array(load_img(path)).astype('float')/255.
        mask = cv.resize(mask, (SIZE,SIZE), cv.INTER_AREA)
        masks[id] = mask[:,:,:1]
    
    # prepare train and test data
    X, y = images[:int(len(images)*0.9)], masks[:int(len(images)*0.9)]
    test_X, test_y = images[int(len(images)*0.9):], masks[int(len(images)*0.9):]

    unet = init_model()
    unet.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('model/segment_mdl.h5',save_best_only=True)]

    with tf.device("/GPU:0"):
        results = unet.fit(
            X, y,
            epochs=100,
            callbacks=callbacks,
            validation_split=0.1,
            batch_size=32)
        
def predict(mdl, path):
    SIZE = 128
    img = img_to_array(load_img(path)).astype('float')/255.
    img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
    real_img = img[np.newaxis,...]
    pred_mask = mdl.predict(real_img).reshape(SIZE, SIZE)
    bw_mask = post_process(pred_mask, 0.5)
    
    rgb_mask = np.zeros([SIZE, SIZE, 3])
    rgb_mask[:,:,0] = bw_mask
    rgb_mask[:,:,1] = bw_mask
    rgb_mask[:,:,2] = bw_mask
    pair_img = np.zeros([SIZE, 2*SIZE, 3])
    pair_img[:, :SIZE, :] = real_img[0]
    pair_img[:, SIZE:, :] = rgb_mask
    pair_img = np.require(pair_img*255, np.uint8, 'C')
    return bw_mask, pair_img

def sortImage(mdl, path):
    SIZE = 128
    image_names = sorted(next(os.walk(path))[-1])
    area_rates = []
    for i in range(len(image_names)):
        img = img_to_array(load_img('{}/{}'.format(path, image_names[i]))).astype('float')/255.
        img = cv.resize(img, (SIZE,SIZE), cv.INTER_AREA)
        real_img = img[np.newaxis,...]
        pred_mask = mdl.predict(real_img).reshape(128,128)
        pred_mask = post_process(pred_mask, 0.4)
        w, h = pred_mask.shape
        area_rates.append(np.round(np.sum(pred_mask)/(w*h)*100, 2))
        
    sortedOrder = np.argsort(area_rates)
    sortedImages = ['{}/{}'.format(path, image_names[i]) for i in sortedOrder]
    sortedAreas = [area_rates[i] for i in sortedOrder]

    return sortedAreas, sortedImages