import numpy as np
import tensorflow as tf
import keras

from data.read_data import Mydata
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.optimizers import adam,SGD,RMSprop
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Activation,LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils#one hot encoding转化

'''
获得训练与测试数据
'''
img_path = '/home/kun/zjw/2/trainval/'
img_path_tes = '/home/kun/zjw/2/test/'
img_width=224
img_height=224
classifiers=3

data = Mydata(img_path,img_path_tes,img_width,img_height)
x_train,y_train,x_test,y_test =data.getnpy()
y_train = np_utils.to_categorical(y_train, classifiers)
y_test = np_utils.to_categorical(y_test, classifiers)
y_test=np.load('y_test.npy')

#图片预处理
#datagen=keras.preprocessing.image.ImageDataGenerator(
 #       featurewise_center=True,
  #      featurewise_std_normalization=True,
   #     horizontal_flip=True,
    #    validation_split=0.2)
#datagen.fit(x_train)
'''
采用imagenet预训练模型
'''
#超参数设置
batch_size=64
epoch=30
lr=1e-4
Optimizer=keras.optimizers.RMSprop(lr=lr)
#Optimizer=keras.optimizers.Adam(lr=lr, beta_1=0.99)
#预训练
inputshape = (img_width, img_height, 3)
#old_model = ResNet50(input_shape=inputshape,weights='imagenet', include_top=True)
old_model = ResNet50(weights='imagenet', include_top=True)
for layer in old_model.layers:
    print(layer.name)
out_put=old_model.get_layer('avg_pool').output
last_output=Dense(classifiers,activation='softmax')(out_put)
mymodel=Model(inputs=old_model.input, outputs=last_output)
for layer in old_model.layers:
    layer.trainable=True
mymodel.compile(optimizer=Optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
saveweights=[ModelCheckpoint('./model_keras/weights.hdf5',save_weights_only=True, mode='max')]
#mymodel.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
 #       steps_per_epoch=len(x_train)/batch_size,epochs=epoch,verbose=1,callbacks=saveweights)
mymodel.fit(x_train, y_train, batch_size=batch_size,
            epochs=epoch, callbacks=saveweights, validation_split=0.2)

out = np.zeros((x_test.shape[0], 3))
out = mymodel.predict(x_test)
y_test_pred = np.argmax(out, axis=1)

score = mymodel.evaluate(x_test, y_test, verbose=2)
print("test score:", score[0])
print("test accuracy:", score[1])

# 导出数据
out_str = list(range(len(y_test_pred)))
for i, item in enumerate(y_test_pred):
    if item == 0:
        out_str[i] = 'A'
    elif item == 1:
        out_str[i] = 'B'
    else:
        out_str[i] = 'C'

out_str = ''.join(out_str)

with open("predict.txt", 'w') as file_out:
    file_out.write(out_str)

