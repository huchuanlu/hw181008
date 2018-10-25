from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import os

from network.myEasyNet import mynet1, mynet2, mynet3, mynet4
from network.vgg import vgg11, vgg13, vgg16, vgg19
from network.resnet import resnet20, resnet32, resnet44, resnet56
from network.xception import XceptionNet
from network.mobileNet import MobileNet
from network.denseNet import DensetNet40_12, DenseNet100_12, DenseNet100_24, DenseNetBC100_12, DenseNetBC250_24, \
    DenseNetBC190_40
from network.resnext import ResNext50, ResNext101
from network.squeezeNet import SqueezeNetA, SqueezeNetB
from network.seNet import SE_Resnet_50, SE_Resnet_101


# 选择使用的gpu
# os.environ['CUDA_VISIBLE_DEVICE'] = '1'

# solver类 更新网络参数
class Solver(object):
    """docstring for Solver"""

    def __init__(self, netname, dataset, common_params, dataset_params):
        super(Solver, self).__init__()

        self.dataset = dataset
        self.learning_rate = common_params['learning_rate']
        self.moment = common_params['moment']
        self.batch_size = common_params['batch_size']
        self.height, self.width = common_params['image_size']

        self.display_step = common_params['display_step']
        self.predict_step = common_params['predict_step']

        self.model_read_path = dataset_params['model_read_path']

        self.netname = netname
        # 路径下的ckpt文件夹
        model_dir = os.path.join(dataset_params['model_path'], self.netname, 'ckpt')
        # 没有模型保存路径，则新建一个
        if not tf.gfile.Exists(model_dir):
            tf.gfile.MakeDirs(model_dir)
        # model_name保存完整路径
        self.model_name = os.path.join(model_dir, 'model.ckpt')

        # 日志文件存放地址
        self.log_dir = os.path.join(dataset_params['model_path'], self.netname, 'log')
        if not tf.gfile.Exists(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)

        # 构建图变量
        self.construct_graph()

    def construct_graph(self):
        # 定义图变量值，不可训练
        self.global_step = tf.Variable(0, trainable=False)

        # 图片和标签的占位符
        self.images = tf.placeholder(tf.float32, (None, self.height, self.width, 3), name='input')
        self.labels = tf.placeholder(tf.int32, None)

        # 训练标志和随机失活率
        self.is_training = tf.placeholder_with_default(False, None, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

        # 读取网络名???????????????????????
        self.net = eval(self.netname)(is_training=self.is_training, keep_prob=self.keep_prob)

        # 定义前向传播过程
        self.predicts, self.softmax_out = self.net.forward(self.images)
        # 定义损失值 由预测值和标签获得
        self.total_loss = self.net.loss(self.predicts, self.labels)
        '''
		指数衰减法设置学习率
		decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
			
			decayed_learning_rate为每一轮优化时使用的学习率
			learning_rate为事先设定的初始学习率
			decay_rate为衰减系数
			decay_steps为衰减速度
			
		staircase(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
		
		例如：learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)     #生成学习率
		每100轮训练后要乘以0.96. 
		'''
        self.learning_rate = self.learning_rate
        #self.learning_rate =  tf.train.exponential_decay(self.learning_rate,self.global_step,50000,0.1,staircase=True)

        # 选择优化器：动量梯度下降
        #optimizer = tf.train.AdamOptimizer(self.learning_rate, self.moment)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
           # self.train_op = optimizer.minimize(self.total_loss,global_step=self.global_step)

        # 计算正确率的方法
        correct_pred = tf.equal(tf.argmax(self.softmax_out, 1, output_type=tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tensorflow的可视化
    # tf.summary.scalar('loss',self.total_loss)
    # tf.summary.scalar('accuracy',self.accuracy)

    def solve(self):
        # Dataset中示例化一个Iterator，然后对Iterator进行迭代
        # 生成一个迭代器，用于遍历所有的数据
        train_iterator=self.dataset['train'].make_one_shot_iterator()
        #train_iterator = self.dataset['train'].make_initializable_iterator()
        # 进行迭代操作：每次列举出下一个数据集
        train_dataset = train_iterator.get_next()
        # 测试集也分batch迭代？？
        test_iterator=self.dataset['test'].make_one_shot_iterator()
        #test_iterator = self.dataset['test'].make_initializable_iterator()
        test_dataset = test_iterator.get_next()

        #train_iterator = train_iterator.initializer
        #test_iterator = test_iterator.initializer

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        # 建立一个 tf.train.Saver() 用来保存, 提取变量
        saver = tf.train.Saver(var_list=var_list)

        config = tf.ConfigProto(allow_soft_placement=False)
        config.gpu_options.allow_growth = False
        init = tf.global_variables_initializer()

        # summary_op = tf.summary.merge_all()

        sess = tf.Session(config=config)
        sess.run(init)
        # summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        '''
		读取保存的模型
		'''
        saver.restore(sess, tf.train.latest_checkpoint(self.model_read_path))

        step = 0
        acc_count = 0
        total_accuracy = 0

        acc_count_tes = 0
        total_accuracy_tes = 0


        #sess.run(test_iterator.initializer)
        #train_iterator.initializer
        #test_iterator.initializer

        coord = tf.train.Coordinator()  # 协同启动的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

        try:
            while True:
                # train_iterator
                #train_iterator=train_iterator.initializer
                #test_iterator=test_iterator.initializer


                #sess.run(train_iterator.initializer)
                #sess.run(test_iterator.initializer)



                #train_iterator=train_iterator.initializer

                images, labels = sess.run(train_dataset)
                #
                # 执行参数优化
                sess.run(self.train_op, feed_dict={self.images: images, self.labels: labels, self.is_training: True,
                                                   self.keep_prob: 0.8})
                #lr = sess.run(self.learning_rate)
                lr = self.learning_rate

                if step % self.display_step == 0:
                    # test_images, test_labels = sess.run(test_dataset)
                    acc, loss = sess.run([self.accuracy, self.total_loss],
                                         feed_dict={self.images: images, self.labels: labels, self.is_training: True,
                                                    self.keep_prob: 0.8})
                    # print(images)
                    total_accuracy += acc
                    acc_count += 1
                    # loss = sess.run(self.total_loss,feed_dict={self.images:images,self.labels:labels,self.is_training:True,self.keep_prob:0.8})
                    print('Iter step:%d learning rate:%.4f loss:%.4f accuracy:%.4f' % (
                    step, lr, loss, total_accuracy / acc_count))
                if step % self.predict_step == 0:
                    # summary_str = sess.run(summary_op,feed_dict={self.images:images,self.labels:labels,self.is_training:True,self.keep_prob:1.0})
                    # summary_writer.add_summary(summary_str,step)
                    # test_dataset这里是一个batch是的，把测试集当成验证集用
                    #sess.run(test_iterator.initializer)
                    test_images, test_labels = sess.run(test_dataset)
                    # print(test_labels)
                    pre, acc = sess.run([self.softmax_out, self.accuracy],
                                        feed_dict={self.images: test_images, self.labels: test_labels,
                                                   self.is_training: False, self.keep_prob: 0.8})
                    # pre1=tf.argmax(pre, 1, output_type=tf.int32)
                    # print(pre1.eval(session=sess))
                    # print(test_labels.T)
                    # print(test_images)
                    # print(pre)
                    total_accuracy_tes += acc
                    acc_count_tes += 1
                    # hahahaha
                    # print(test_labels.shape)
                    print('test acc:%.4f' % (total_accuracy_tes / acc_count_tes))

                # 10轮训练保存一次模型
                if step % 20 == 0:
                    # 保存,路径在model_name中
                    # max_to_keep参数定义saver()将自动保存的最近n个ckpt文件，
                    # 默认n=5，即保存最近的5个检查点ckpt文件。若n=0或者None，则保存所有的ckpt文件。
                    saver.save(sess, self.model_name, global_step=step)
                step += 1


        # 直到读不到数据
        except tf.errors.OutOfRangeError:
            #coord.request_stop()  # 停止所有的线程
            #coord.join(threads)
            print("finish training !")
        sess.close()
