from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from skimage import io,transform

import os
import tensorflow as tf 
import config.cfg as cfg
import glob
import numpy as np

#图片尺寸要和模型中统一
_HEIGHT = 100
_WIDTH = 100
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1#多了一个bit是标签吗？
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 2000,#1165,#50000
    'validation': 1000,
}


#从路径读取二进制文件信息
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  #确定有此文件夹
  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record,is_training):
    #解析器
    record_vector = tf.decode_raw(raw_record,tf.uint8)
    '''
    decode_raw操作可以讲一个字符串转换为一个uint8的张量
    '''

    #读出标签
    label = tf.cast(record_vector[0],tf.int32)
    #读出所有的图
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],[_NUM_CHANNELS,_HEIGHT,_WIDTH])

    #用于改变某个张量的数据类型 tf.cast
    '''
    tf.transpose转置
    tf.transpose的第二个参数perm=[0,1,2],
    0代表三维数组的高（即为二维数组的个数），1代表二维数组的行，2代表二维数组的列。
    '''
    image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)
    '''
    图像预处理
    '''
    image = preprocess_image(image,is_training)
    return image,label


def preprocess_image(image, is_training):
    if is_training:
        #预处理 resize 定的稍微大些，为后面的随机剪裁做准备
        image = tf.image.resize_image_with_crop_or_pad(image,_HEIGHT+8,_WIDTH+8)
        #随机剪裁
        image = tf.random_crop(image,[_HEIGHT,_WIDTH,_NUM_CHANNELS])
        #随机左右翻转
        image = tf.image.random_flip_left_right(image)
    #图片归一化
    image = tf.image.per_image_standardization(image)
    return image


def my_preprocess_image(image, is_training):
    if is_training:

        #预处理 resize 定的稍微大些，为后面的随机剪裁做准备
        image = tf.image.resize_image_with_crop_or_pad(image,_HEIGHT+8,_WIDTH+8)
        #随机剪裁
        image = tf.random_crop(image,[_HEIGHT,_WIDTH,_NUM_CHANNELS])
        #随机左右翻转
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT, _WIDTH)
    #图片归一化
    image = tf.image.per_image_standardization(image)
    return image

def input_fn(is_training, common_params,dataset_params):
    data_dir = dataset_params['data_path']
    batch_size = common_params['batch_size']
    num_epochs = common_params['num_epochs']

    '''
    读出文件夹中的所有文件名 函数
    文件是几个包含图片标签信息的二进制文件
    '''
    filenames = get_filenames(is_training,data_dir)

    '''
    从二进制文件中读取固定长度纪录， 
    '''
    dataset = tf.data.FixedLengthRecordDataset(filenames,_RECORD_BYTES)
    #将数据全部读入

    # 读取一部分数据到内存中，“预取”数据
    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        # 训练样本数量的乱序
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    dataset = dataset.repeat(num_epochs)#复制epoch份
    # 数据统一转换处理 自定义dataset转换链接
    '''
    tf.data API提供了tf.contrib.data.map_and_batch转换，
    它可以用效地将map转换和batch转换相“混合（fuse）
    '''
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda value: parse_record(value, is_training),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset



def my_input_fn2(is_training, common_params, dataset_params):
    # 选则测试or训练 先指定路径
    if is_training:
        data_dir = dataset_params['data_path']
    else:
        data_dir = dataset_params['data_path1']

    # 设置参数
    batch_size = common_params['batch_size']
    num_epochs = common_params['num_epochs']

    if is_training:
        cate = [data_dir + x for x in os.listdir(data_dir) if os.path.isfile(data_dir + x)]
        imgspaths = []
        labels = []
        print('reading the train images...')
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder):
                imgspaths.append(im)
                labels.append(ord(im[-14]) - 65)
        print('train images read')

        # 建立 Queue
        imgspath, label = tf.train.slice_input_producer([imgspaths, labels], shuffle=True)

        # 读取图片，并进行解码
        image = tf.read_file(imgspath)
        image = tf.image.decode_jpeg(image, channels=_NUM_CHANNELS)

        image=my_preprocess_image(image, is_training)


    else:
        name_list = [x for x in os.listdir(data_dir)]
        name_list.sort(key=lambda x: int(x[:-4]))
        cate = [data_dir + item for item in name_list]
        imgspaths = []
        labels = []
        labels = np.load('y_test.npy')
        # print(len(cate))
        # print(labels.shape)
        print('reading the test images...')
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder):
                # print('reading the images:%s'%(im))
                imgspaths.append(im)
            # labels.append(ord(im[-14]) - 65)
        print('test images read')
        y_test = labels.reshape(1, -1)
        N, D = labels.shape
        y_tes = np.zeros(N * D)

        y_tes = labels.T[0][:].astype(np.int32)
        labels = labels.T[0][:].astype(np.int32)

        # 建立 Queue  只有这样才能预处理图片
        imgspath, label = tf.train.slice_input_producer([imgspaths, y_tes], shuffle=False,num_epochs=1)

        # 读取图片，并进行解码
        #threads = tf.train.start_queue_runners(sess=sess)
        #image = tf.read_file(imgspath)
        image_reader = tf.WholeFileReader()
        key, image = image_reader.read(imgspath)
        image = tf.image.decode_jpg(image, channels=_NUM_CHANNELS)

        image=my_preprocess_image(image, is_training)


    #with tf.Session() as sess:
        #tf.local_variables_initializer().run()
        #threads = tf.train.start_queue_runners(sess=sess)
        #image=image.eval(session=sess)
    #print(image)
    data, label = np.asarray(image, np.float32), np.asarray(labels, np.int32)


    dataset = tf.data.Dataset.from_tensor_slices((data, label))  # 构建Dataset
    #dataset = tf.data.Dataset.from_tensors((image, label))

    # 测试集要设置batch?
    dataset = dataset.batch(batch_size)  # 设置Dataset的batch大小

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    # 复制epoch份
    dataset = dataset.repeat(num_epochs)

    # 自动预读取
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset





def my_input_fn1(is_training, common_params, dataset_params):
    # 选则测试or训练 先指定路径
    if is_training:
        data_dir = dataset_params['data_path']
    else:
        data_dir = dataset_params['data_path1']

    # 设置参数
    batch_size = common_params['batch_size']
    num_epochs = common_params['num_epochs']

    if is_training:
        cate = [data_dir + x for x in os.listdir(data_dir) if os.path.isfile(data_dir + x)]
        imgspaths = []
        labels = []
        print('reading the train images...')
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder):
                imgspaths.append(im)
                labels.append(ord(im[-14]) - 65)
        print('train images read')

        # 建立 Queue
        imgspath, label = tf.train.slice_input_producer([imgspaths, labels], shuffle=True)

        # 读取图片，并进行解码

        #image_reader = tf.WholeFileReader()
        #key, image = image_reader.read(imgspath)

        image = tf.read_file(imgspath)
        image = tf.image.decode_jpeg(image, channels=_NUM_CHANNELS)

        image=my_preprocess_image(image, is_training)

        # with tf.Session() as sess:
        #     threads = tf.train.start_queue_runners(sess=sess)
        #     image=image.eval(session=sess)
        # pass


    else:
        name_list = [x for x in os.listdir(data_dir)]
        name_list.sort(key=lambda x: int(x[:-4]))
        cate = [data_dir + item for item in name_list]
        imgspaths = []
        labels = []

        labels = np.load('y_test.npy')
        # print(len(cate))
        # print(labels.shape)
        print('reading the test images...')
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder):
                # print('reading the images:%s'%(im))
                imgspaths.append(im)
            # labels.append(ord(im[-14]) - 65)
        print('test images read')
        y_test = labels.reshape(1, -1)
        N, D = labels.shape
        y_tes = np.zeros(N * D)
        y_tes = labels.T[0][:].astype(np.int32)
        #print(imgspaths[5])
        # 建立 Queue
        imgspath, label = tf.train.slice_input_producer([imgspaths, y_tes], shuffle=True)
        #print(imgspath.shape)

        # 读取图片，并进行解码

        #image_reader = tf.WholeFileReader()
        #key, image = image_reader.read(imgspath)

        image = tf.read_file(imgspath)
        image = tf.image.decode_jpeg(image, channels=_NUM_CHANNELS)

        #print(image.shape)

        image=my_preprocess_image(image, is_training)


    # 打乱顺序
    # num_example = data.shape[0]
    # arr = np.arange(num_example)
    # np.random.shuffle(arr)
    # data = data[arr]
    # label = label[arr]
    # data label为numpy array类型

    '''
    使用tf.data.Dataset构建数据集,从张量中直接读取数据

    tf.data.Dataset.from_tensor_slices()函数会对tensor和numpy array的处理一视同仁，
    所以该函数既可以使用tensor参数，也可以直接使用numpy array作参数

    Dataset 不知道自己包含多少条目
    '''
    #dataset = tf.data.Dataset.from_tensor_slices((data, label))  # 构建Dataset
    dataset = tf.data.Dataset.from_tensors((image, label))
    #tf.data.Dataset.from_tensor_slices((image, label))

    # 测试集要设置batch?
    dataset = dataset.batch(batch_size)  # 设置Dataset的batch大小

    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    # 复制epoch份
    dataset = dataset.repeat(num_epochs)

    #kkk=image,label

    #dataset = dataset.apply(tf.contrib.data.map_and_batch(
        #kkk,
        #batch_size=batch_size,
        #num_parallel_batches=1,
        #drop_remainder=False))

    # 自动预读取
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset



'''
我们通过TensorFlow提供的tensor操作来读取数据，并基于此，构建Dataset
'''
def my_input_fn(is_training, common_params,dataset_params):

    #设置参数
    batch_size = common_params['batch_size']
    num_epochs = common_params['num_epochs']

    imgs=np.load('x_train.npy')
    x_tra = imgs.T[0][:].astype(np.int32)
    labels=np.load('y_train.npy')
    y_tes = labels.T[0][:].astype(np.int32)

    data,label=np.asarray(x_tra,np.float32),np.asarray(y_tes,np.int32)
    data, label = tf.convert_to_tensor(data),tf.convert_to_tensor(label)

    dataset = tf.data.Dataset.from_tensor_slices((data, label))#构建Dataset

    #测试集要设置batch?
    dataset = dataset.batch(batch_size)#设置Dataset的batch大小

    #'''
    #读取一部分数据到内存中，“预取”数据
    #'''
    #dataset = dataset.prefetch(buffer_size=batch_size)
    #打乱顺序
    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    # 复制epoch份
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def cifar_dataset(common_params,dataset_params):
    train_dataset = input_fn(True,common_params,dataset_params)
    test_dataset = input_fn(False,common_params,dataset_params)
    dataset = {
        'train':train_dataset,
        'test':test_dataset
    }
    return dataset

#输入公共参数和数据参数
'''
返回测试数据、训练数据的字典，包括标签
'''
def my_dataset(common_params,dataset_params):
    #按测试和训练数据读入
    train_dataset = my_input_fn(True,common_params,dataset_params)
    #test_dataset = my_input_fn(False,common_params,dataset_params)
    #放入dataset字典
    dataset = {
        'train':train_dataset,
        #'test':test_dataset
    }
    return dataset

def input_fn_test(dataset_params):
    data_dir = dataset_params['data_path']
    batch_size = _NUM_IMAGES['validation']
    filenames = get_filenames(False,data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames,_RECORD_BYTES)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        lambda value: parse_record(value, False),
        batch_size=batch_size,
        num_parallel_batches=1,
        drop_remainder=False))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def my_input_fn_test(dataset_params):
    data_dir = dataset_params['data_path1']
    batch_size = _NUM_IMAGES['validation']

    name_list = [x for x in os.listdir(data_dir)]
    name_list.sort(key=lambda x: int(x[:-4]))
    cate = [data_dir + item for item in name_list]
    imgs = []
    labels = []
    labels = np.load('y_test.npy')
    # print(len(cate))
    # print(labels.shape)
    print('reading the test images...')
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder):
            # print('reading the images:%s'%(im))
            img = io.imread(im)
            img = transform.resize(img, (_WIDTH, _HEIGHT))
            imgs.append(img)
    # labels.append(ord(im[-14]) - 65)
    print('test images read')
    y_test = labels.reshape(1, -1)
    N, D = labels.shape
    y_tes = np.zeros(N * D)
    y_tes = labels.T[0][:]
    data, label = np.asarray(imgs, np.float32), np.asarray(y_tes, np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((data, label))  # 构建Dataset
    dataset = dataset.batch(batch_size)  # 设置Dataset的batch大小

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

def cifar_dataset_test(dataset_params):
    dataset = input_fn_test(dataset_params)
    return dataset

def my_dataset_test(dataset_params):
    dataset = my_input_fn_test(dataset_params)
    return dataset

if __name__=='__main__':
    pass
    #dataset = cifar_dataset_test(cfg.dataset_params)
    #iterator = dataset.make_one_shot_iterator()
    #next_element = iterator.get_next()
    #sess = tf.Session()
    #images,labels = sess.run(next_element)
    #print(images.shape)
    #print(dataset)