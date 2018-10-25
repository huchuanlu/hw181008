from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

cfg = {
    'mynet5': [32, 'M', 64, 'M', 128, 'M', 128, 'M'],
    'mynet6': [],
    'mynet7': [],
    'mynet8': []
}

#自定义网络
class MydetectionNet(object):

    def __init__(self, mynetname, is_training, keep_prob=0.5, pre_point=3):
        super(MydetectionNet, self).__init__()
        self.mynetname = mynetname
        self.pre_point = pre_point

        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-3)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.pool_num = 0
        self.conv_num = 0
        self.is_training = is_training

        self.keep_prob = keep_prob

    def forward(self, input):
        out = self.make_layer(input, cfg[self.mynetname])
        out = tf.layers.flatten(out, name='flatten')
        predicts = tf.layers.dense(out, units=self.pre_point, kernel_initializer=self.initializer,
                                   kernel_regularizer=self.regularizer, name='fc_1')
        #softmax_out = tf.nn.softmax(predicts, name='output')
        return predicts

    def conv2d(self, inputs, out_channel):
        inputs = tf.layers.conv2d(inputs, filters=out_channel, kernel_size=3, padding='same',
                                  kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                  name='conv_' + str(self.conv_num))
        #inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name='bn_' + str(self.conv_num))
        self.conv_num += 1
        return tf.nn.relu(inputs)

    def make_layer(self, inputs, netparam):
        for param in netparam:
            if param == 'M':
                inputs = tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding='same',
                                                 name='pool_' + str(self.pool_num))
                self.pool_num += 1
            else:
                inputs = self.conv2d(inputs, param)
        inputs = tf.layers.average_pooling2d(inputs, pool_size=1, strides=1)
        return inputs

    def loss(self, predicts, labels):
        losses = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, predicts))
        l2_reg = tf.losses.get_regularization_losses()
        losses += tf.add_n(l2_reg)
        return losses


def mynet5(is_training=True, keep_prob=0.5):
    net = MydetectionNet(mynetname='mynet5', is_training=is_training, keep_prob=keep_prob)
    return net


def mynet2(is_training=True, keep_prob=0.5):
    net = MydetectionNet(mynetname='mynet6', is_training=is_training, keep_prob=keep_prob)
    return net


def mynet6(is_training=True, keep_prob=0.5):
    net = MydetectionNet(mynetname='mynet7', is_training=is_training, keep_prob=keep_prob)
    return net


def mynet7(is_training=True, keep_prob=0.5):
    net = MydetectionNet(mynetname='mynet8', is_training=is_training, keep_prob=keep_prob)
    return net
