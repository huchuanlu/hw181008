from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config.cfg as cfg 
from solver.solver_cifar import Solver
from data.dataset import my_dataset

import argparse #命令解析模块
import tensorflow as tf 

'''
在此处用default形式先定义好参数
'''
parser = argparse.ArgumentParser()#创建一个命令行对象
#增加命令行
'''
第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
'''
#学习率设置过大会导致正确率上不去
parser.add_argument('--lr',type=float,default=0.01,help='learning_rate')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--moment',type=float,default=0.9,help='sovler moment')
parser.add_argument('--display_step',type=int,default=5,help='show train display')
parser.add_argument('--num_epochs',type=int,default=150,help='train epochs')
parser.add_argument('--predict_step',type=int,default=50,help='predict step')
parser.add_argument('-n','--net',type=str,default='vgg11',choices=cfg.net_style,help='net style')


def main(_):
	common_params = cfg.merge_params(FLAGS)#从解析出的命令中读取参数(公共的)
	print(FLAGS.net)
	print(common_params)
	net_name = FLAGS.net#读网络名
	'''
	读取数据
	'''
	dataset = my_dataset(common_params,cfg.dataset_params)

	'''
	建立求解器类
	'''
	solver = Solver(net_name,dataset,cfg.common_params,cfg.dataset_params)
    #求解
	solver.solve()

if __name__=='__main__':
	'''
	解析参数，等于flags,unparsed=parse.parse_known_args(sys.argv[1:])
	其中参数sys.argv[1:]是命令行语句中从第一个到最后，可以不写，默认就是
	flags同parse_args()输出结果相同，都是命名空间
	
	arg=parse_args（）和 flags, unparsed=parse_known_args（）用法基本一样，都是解析输出参数的
	若使用：arg=parse.parse_args(sys.argv[1:])语句代替上述，并在ubuntu下输入命令行
	python train.py --lr=20 --net=vgg16
	则输出的arg为namespace空间（记录对象和对象名字对应关系的空间，
	现今 Python 的大部分命名空间是通过字典来实现的，
	也即一个命名空间就是名字到对象的映射，标识符是键，对象则是值），
	结果是Namespace(lr=20,net=vgg16)
	'''
	FLAGS,unknown = parser.parse_known_args()
	'''
	把日志设置为INFO级别
	
	tensorflow使用五级日志：DEBUG, INFO, WARN, ERROR, and FATAL
	使用tf.logging模块，位置在tensorflow\python\platform\tf_logging.py，
	这个模块基于python3的logging模块创建。
	当设置INFO级别后，tf.contrib.learn会自动每百步后输出损失度量数据到标准输出。
	'''
	tf.logging.set_verbosity(tf.logging.INFO)
	'''
	函数是用来处理flag解析，然后执行main函数
	
	tensorflow的程序中,在main函数下,一般都是使用tf.app.run()来启动
	flag解析：执行启动之前要解析以tensorflow方式定义的变量
	'''
	tf.app.run(main)