from __future__ import print_function
import numpy as np
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
import os

from keras.models import Model
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from tools.getiou import Result
from keras.preprocessing import image
from scipy.misc import imread
from xml.etree import ElementTree as ET
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

#使用GPU
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

#超参数设置
epochs = 120
batch_size = 64
base_lr = 1e-4

#读取数据
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

#预训练模型设置 使用VGG19
height=224
width=224
inputshape = (width, height, 3)
base_model = VGG19(input_shape=inputshape,
                         weights='imagenet',
                         include_top=True)

for layer in base_model.layers:
    print(layer.name)
#VGG19_out = base_model.output
#VGG19_out = base_model.get_layer('fc2').output
VGG19_out = base_model.get_layer('fc1').output

#定义base_model之后的结构
# conv_2d = Conv2D(1024, 3, padding='same', activation='relu')(ResNet50_out)
# conv_2d1 = Conv2D(256, 5, padding='same', activation='relu')(conv_2d)
# global_avgpooling = GlobalAveragePooling2D()(conv_2d1)
#dense1 = Dense(4096, activation='relu')(ResNet50_out)
#dense2 = Dense(128, activation='relu')(dense1)
predictions = Dense(4, activation='linear')(VGG19_out)

#整合网络 允许base_model更新
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True

#学习率更新
def schedule(epoch, decay=0.98):
    return base_lr * decay**(epoch)

#保存模型参数
callbacks = [keras.callbacks.ModelCheckpoint(
    './model_data/weights_best_VGG19.hdf5',
    verbose=1, save_weights_only=True, mode='max')]

# 优化器与损失函数
optim = keras.optimizers.Adam(lr=base_lr, beta_1=0.99)
model.compile(optimizer=optim, loss='mse', metrics=['mae'])

# 训练
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=1, callbacks=callbacks,
          validation_split=0.2)

#预测训练集
results = model.predict(x_train, verbose=1)

#训练正确率
re=Result(results,y_train)
iou=re.iou()
print(iou)

'''
使用训练完的模型进行测试
'''
box_testpath = './test/box/'
img_testpath = './test/img/'

filenames=[]
for filename in os.listdir(img_testpath):
    filenames.append(filename[:-4])
_height = 224;
_width = 224;
img_in = np.zeros((len(filenames), _width, _height, 3))
for i in range(len(filenames)):
    img = Image.open(img_testpath + filenames[i] + '.jpg')
    img = img.resize((_width, _height))
    img_in[i] = np.array(img)

model.load_weights('./model_data/weights_best_VGG19.hdf5', by_name=True)
prediction = model.predict(img_in, batch_size=1, verbose=1)

# 保存结果到xml文件
for i, img_name in enumerate(filenames):
    img_read = imread(img_testpath + img_name + '.jpg')

    pre_xmin = prediction[i][0]
    pre_ymin = prediction[i][1]
    pre_xmax = prediction[i][2]
    pre_ymax = prediction[i][3]

    annotation = ET.Element("prediction")
    bnbox = ET.SubElement(annotation,"bndbox")
    xmin = ET.SubElement(bnbox, "xmin")
    ymin = ET.SubElement(bnbox, "ymin")
    xmax = ET.SubElement(bnbox, "xmax")
    ymax = ET.SubElement(bnbox, "ymax")
    conf = ET.SubElement(bnbox, "conf")

    # 限幅
    if pre_xmax > img_read.shape[1]:
        pre_xmax = img_read.shape[1]
    if pre_ymax > img_read.shape[0]:
        pre_ymax = img_read.shape[0]

    xmin.text = str(int(round(pre_xmin)))
    ymin.text = str(int(round(pre_ymin)))
    xmax.text = str(int(round(pre_xmax)))
    ymax.text = str(int(round(pre_ymax)))
    conf.text = str(float(1))

    tree = ET.ElementTree(annotation)
    tree.write(box_testpath + img_name[:-3] + 'xml')