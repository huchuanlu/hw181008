# # 指定第一块GPU可用
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# sess = tf.Session(config=config)
#
# KTF.set_session(sess)

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Activation,LeakyReLU
from keras.optimizers import RMSprop,adam,SGD,adamax
import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np

#np.random.seed(1671)  # for reproducibility
X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# 除以256速度变慢,需要更多的迭代
X_train = X_train
X_test = X_test
# network and training
NB_EPOCH = 45
BATCH_SIZE = 64
VERBOSE = 2
NB_CLASSES = 3   # number of outputs = number of digits
# 为什么SGD要比adam效果要好
#OPTIMIZER = SGD(lr=0.0001, decay=1e-6, momentum=0.9)  #  optimizer, explainedin this chapter
OPTIMIZER = adamax()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0

print(X_train.shape[0], 'train samples')
#Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_train = y_train
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
base_model =ResNet50(weights='imagenet', include_top=False)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# model.summary()
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

history = model.fit(X_train, Y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_train, Y_train, verbose=VERBOSE)

print("\nTest score:", score[0])
print('Test accuracy:', score[1])

score1 = model.evaluate(X_test, Y_test, verbose=VERBOSE)
preds = model.predict(X_test)
preds = np.argmax(preds,axis=-1)
print("\nTest score:", score1[0])
print('Test accuracy:', score1[1])
print(preds)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()