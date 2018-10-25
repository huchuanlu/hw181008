import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import random
import keras.backend as K
import keras.losses
from keras.models import Model,load_model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense,Flatten,MaxPooling2D,Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
#%matplotlib inlile

imsize = 224
img_dir = 'trainval/img/'
imgs = os.listdir(img_dir)
num_total = len(imgs)
X = np.zeros((num_total,imsize,imsize,3))
Y = np.zeros((num_total,4))
LW = np.zeros((num_total,2))

# 这里是对img和box先做一个预处理
for it in range(num_total):
    img = Image.open('trainval/img/' + imgs[it])
    l, w, _ = np.array(img).shape
    LW[it] = np.array([l, w])
    img = img.resize((imsize, imsize))
    img = np.array(img)
    X[it] = img / 255
    img_name = imgs[it][:-4]
    ann_name = img_name + '.xml'
    ann = ET.parse('trainval/box/' + ann_name)
    box = [int(ann.find(tag).text) - 1
           for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
    xmin, ymin, xmax, ymax = box

    xmin_norm, xmax_norm = [xmin / w, xmax / w]
    ymin_norm, ymax_norm = [ymin / l, ymax / l]

    x_c = (xmin_norm + xmax_norm) / 2 * imsize
    y_c = (ymin_norm + ymax_norm) / 2 * imsize

    bbox_w = (xmax_norm - xmin_norm) * imsize
    bbox_l = (ymax_norm - ymin_norm) * imsize
    Y[it] = [x_c, y_c, bbox_w, bbox_l]

np.save("X.npy", X)
np.save("Y.npy", Y)

X = np.load('X.npy')
Y = np.load('Y.npy')

random.seed(10)
permutation = np.random.permutation(Y.shape[0])
X = X[permutation, :]
Y = Y[permutation]
X_train = X
Y_train = Y

HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)

keras.losses.smoothL1 = smoothL1

def Treatment(nb_classes=4, img_rows=imsize, img_cols=imsize, RGB=True):
    color = 3 if RGB else 1
    base_model = VGG19(weights='imagenet', include_top=True, pooling=None, input_shape=(img_rows, img_cols, color))
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.get_layer('fc2').output
    x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=smoothL1, optimizer=adam)
    return model

def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)
model = Treatment()
model.fit(X_train, Y_train, batch_size=16, epochs=25,callbacks=[reduce_lr])

model.save('model')
