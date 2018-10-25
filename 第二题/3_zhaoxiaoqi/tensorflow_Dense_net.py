# 考试第二题
# 没有使用dropout 只用了bn ,就有抗过拟合的作用
# Densenet会节省参数内存。
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #one-hot表示的数据集的标签是one-hot形式
X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
print(y_train.shape)
X_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# 定义网络超参数

learning_rate = 0.001
training_iters = 20000
batch_size = 32
display_step = 20

# 定义网络参数
n_input = 784 # 输入的维度
n_classes = 3 # 标签的维度
dropout = 0.8 # Dropout 的概率

# 占位符输入
x = tf.placeholder(tf.float32, [None, 224,224,3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def separable_conv(input,filters,kernel_size,layer_name="conv",padding='SAME'):
    N = input.shape[3].value
    layers_concat = []
    for i in range(N):
        input_x = conv_layer(input[:,:,:,i],1,[3,3])
        layers_concat.append(input_x)
    x = tf.concat(layers_concat, axis=3)
    net = conv_layer(x,32,[1,1])
    return net


def conv_layer(input,filters,kernel_size,stride=[1,1],layer_name="conv",padding='SAME'):
    net = slim.conv2d(input,filters,kernel_size,stride=stride,padding=padding)
    return net




def flatten(x):
    """
    Input:
    - TensorFlow Tensor of shape (N, D1, ..., DM)

    Output:
    - TensorFlow Tensor of shape (N, D1 * ... * DM)
    """
    N = tf.shape(x)[0]
    N1 = x.shape[1].value
    N2 = x.shape[2].value
    N3 = x.shape[3].value
    M = N1*N2*N3
    M = tf.cast(M,dtype=tf.int32)
    #x = tf.reshape(x, (N, -1)) #为什么用这种方法，下面就提取不出x的维度值
    x = tf.reshape(x,(N,M))
    return x

#  对于class 调用自己的内部的函数或者变量,都要加self.
class DenseNet():
    def __init__(self,x,nb_blocks,filters):
        self.nb_blocks = nb_blocks
        self.filters  = filters
        self.model = self.build_model(x)
        #self.sess = sess
#  注意写tensorflow的框架时，从小到大。从里到外
    def bottleneck_layer(self,x,scope):
        #BN->relu->conv1*1->BN->Relu->conv3*3
        with tf.name_scope(scope):
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x,self.filters,kernel_size=(1,1),layer_name=scope+'_conv1')
            x = slim.batch_norm(x)
            x = tf.nn.relu(x)
            x = conv_layer(x,self.filters,kernel_size=(3,3),layer_name=scope+'_conv2')
            return x


    def transition_layre(self,x,scope):
        #BN->CONV1*1->avg_pool2d
        with tf.name_scope(scope):
            x = slim.batch_norm(x)
            x = conv_layer(x,self.filters,kernel_size=(1,1),layer_name=scope+'_conv1')
            x = slim.avg_pool2d(x,2,padding='SAME')
            return x

#   单独的一个denseblock nb_layers表示bottleneck块的个数
    def dense_block(self,input_x,nb_layers,layer_name):
        with tf.name_scope(layer_name):
            layers_concat = []
            layers_concat.append(input_x)
            x = self.bottleneck_layer(input_x,layer_name+'_bottleN_'+str(0))
            layers_concat.append(x)
            for i in range(nb_layers):
                x = tf.concat(layers_concat,axis=3)
                x = self.bottleneck_layer(x,layer_name+'_bottleN_'+str(i+1))
                layers_concat.append(x)
            return x

    def build_model(self,input_x):
        x = tf.reshape(input_x,shape=[-1,224,224,3])
        x = conv_layer(x,self.filters,[7,7],padding='VALID',layer_name='conv0')
        x = slim.max_pool2d(x,(2,2))
        #4个blocks 每个blocks又有4个nb_layers
        for i in range(self.nb_blocks):
            print(i)
            x = self.dense_block(x,4,'dense_'+str(i))
            x = self.transition_layre(x,'trans_'+str(i))
        return x


# 构建模型
score = DenseNet(x,nb_blocks=4,filters=32)
score = score.build_model(x)
score = slim.avg_pool2d(score,(1,1))
score = flatten(score)  #确实已经有作用了。但是为什么，shape显示不出
N = score.shape[1].value
w = tf.Variable(tf.random_normal([N,n_classes]))
pred = tf.matmul(score,w)
# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.initialize_all_variables()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 0
    i = 0
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 获取批数据

        if i*batch_size > 1165:
            batch_xs = X_train[(i-1)*batch_size:1165]
            batch_ys = y_train[(i-1)*batch_size:1165]
            i = 0
        batch_xs = X_train[i*batch_size:i*batch_size+batch_size]
        batch_ys = y_train[i*batch_size:i*batch_size+batch_size]
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
        i += 1
    print("Optimization Finished!")
    # 计算测试精度
    k = 0
    acc = 0
    while k * batch_size < 1165:
        batch_xs = X_train[k * batch_size:k * batch_size + batch_size]
        batch_ys = y_train[k * batch_size:k * batch_size + batch_size]
        acc += sess.run(accuracy, feed_dict={x:batch_xs, y: batch_ys, keep_prob: 1.})
        k += 1
        if k * batch_size > 1165:
            batch_xs = X_train[(k - 1) * batch_size:1165]
            batch_ys = y_train[(k - 1) * batch_size:1165]
            acc += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            acc = acc/(k+1)


 #   print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_train, y:y_train, keep_prob: 1.}))
    print("Testing Accuracy:",acc)
    k = 0
    acc = 0
    while k * batch_size < 540:
        batch_xs = X_test[k * batch_size:k * batch_size + batch_size]
        batch_ys = y_test[k * batch_size:k * batch_size + batch_size]
        acc += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        k += 1
        if k * batch_size > 540:
            batch_xs = X_train[(k - 1) * batch_size:540]
            batch_ys = y_train[(k - 1) * batch_size:540]
            acc += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            acc = acc / (k + 1)

    #   print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:X_train, y:y_train, keep_prob: 1.}))
    print("Testing Accuracy:", acc)
    #print("testing accuary_test",sess.run(accuracy,feed_dict={x:X_test,y:y_test,keep_prob:1.}))
