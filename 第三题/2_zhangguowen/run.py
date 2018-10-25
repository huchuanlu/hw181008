from PIL import Image
import numpy as np
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
from xml.dom.minidom import Document

imsize = 224

img_dir = 'test/img/'
imgs = os.listdir(img_dir)
X_test = np.zeros((len(imgs),imsize,imsize,3))
LW_test = np.zeros((len(imgs),2))


HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)
keras.losses.smoothL1 = smoothL1


model = load_model('model')

for it in range(len(imgs)):
    img = Image.open('test/img/'+imgs[it])
    l,w,_ = np.array(img).shape
    LW_test[it] = np.array([l,w])
    img = img.resize((imsize,imsize))
    img = np.array(img)
    X_test[it] = img/255

np.save("X_test.npy",X_test)
X_test = np.load('X_test.npy')

for it in range(len(imgs)):
    bbox = model.predict(X_test[it:it+1])
    bbox = bbox.squeeze()
    x,y,w,l = bbox
    x_min,y_min,x_max,y_max = [x-w/2,y-l/2,x+w/2,y+l/2]
    x_min = x_min/imsize * LW_test[it,1]
    x_max = x_max/imsize * LW_test[it,1]
    y_min = y_min/imsize * LW_test[it,0]
    y_max = y_max/imsize * LW_test[it,0]
    img_name = imgs[it][:-4]
    box = [x_min,y_min,x_max,y_max]
    conf = 1
    result = ET.Element("prediction")
    ann_box = ET.SubElement(result,"bndbox")
    ET.SubElement(ann_box, "xmin").text = str(int(box[0]))
    ET.SubElement(ann_box, "ymin").text = str(int(box[1]))
    ET.SubElement(ann_box, "xmax").text = str(int(box[2]))
    ET.SubElement(ann_box, "ymax").text = str(int(box[3]))
    ET.SubElement(ann_box, "conf").text = '%.4f'%conf
    tree = ET.ElementTree(result)
    tree.write('test/box/'+str(img_name)+'.xml')