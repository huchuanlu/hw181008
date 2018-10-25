import os 

#数据参数主要包含数据和模型路径
dataset_params = {
	'data_path':'/home/jill/研0考试/181008/zjw/2_1/trainval/',
    'data_path1':'/home/jill/研0考试/181008/zjw/2_1/test/',
	'model_path':'./model/train',
    'model_read_path':'model/train/mynet1/ckpt/'

}

common_params = {
	'batch_size':20,
	'image_size':(100,100),
	'learning_rate':0.1,
	'moment':0.9,
	'display_step':5,
	'num_epochs':200,
	'predict_step':5
}


graph_node = {
	'input':'input:0',
	'is_training':'is_training:0',
	'keep_prob':'keep_prob:0',
	'output':'output:0'
}

net_style = ['vgg11','vgg13','vgg16','vgg19',
			'resnet20','resnet32','resnet44','resnet56',
			'XceptionNet',
			'MobileNet',
			'DensetNet40_12','DenseNet100_12','DenseNet100_24','DenseNetBC100_12','DenseNetBC250_24','DenseNetBC190_40',
			'ResNext50','ResNext101',
			'SqueezeNetA','SqueezeNetB',
			'SE_Resnet_50','SE_Resnet_101','mynet1']


def merge_params(FLAGS):
	common_params['batch_size'] = FLAGS.batch_size
	common_params['learning_rate'] = FLAGS.lr
	common_params['moment'] = FLAGS.moment
	common_params['display_step'] = FLAGS.display_step
	common_params['num_epochs'] = FLAGS.num_epochs
	common_params['predict_step'] = FLAGS.predict_step
	return common_params