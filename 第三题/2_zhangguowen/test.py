import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import random
import keras.backend as K
import keras.losses
from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense,Flatten,MaxPooling2D,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)
keras.losses.smoothL1 = smoothL1


X = np.load('X.npy')
Y = np.load('Y.npy')

X_val = X[:500]
Y_val = Y[:500]
model = load_model('model')



def iou(bbox,gt):
    iou = 0
    x,y,w,l = bbox
    x_gt,y_gt,w_gt,l_gt = gt
    x1,y1,x2,y2 = [x-w/2,y-l/2,x+w/2,y+l/2]
    x1_gt,y1_gt,x2_gt,y2_gt = [x_gt-w_gt/2,y_gt-l_gt/2,x_gt+w_gt/2,y_gt+l_gt/2]
    xi1 = max(x1,x1_gt)
    xi2 = min(x2,x2_gt)
    yi1 = max(y1,y1_gt)
    yi2 = min(y2,y2_gt)
    area_i = max(0,(xi2-xi1))*max(0,(yi2-yi1))
    area1 = w*l
    area2 = w_gt*l_gt
    area_u = area1+area2-area_i
    iou = area_i/area_u
    return iou

threshold = 0.5
count = 0
num_val = 200
for i in range(num_val):
    bbox = model.predict(X_val[i:i+1])
    bbox = bbox.squeeze()
    GT = Y_val[i]
    tmp = iou(bbox,GT)
    if tmp>threshold:
        count=count+1
print("准确度为：",count/num_val)