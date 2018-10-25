import matplotlib.pyplot as plt
import numpy as np
#from compiler.ast import flatten
# 梯度更新函数
def gradientAscent(feature,label_data,k,maxCycle,alpha):
        #input: feature_data(mat):特征
            #     label_data(mat):标签
            #     k(int):类别的个数
            #     maxCycle(int):最大迭代次数
            #     alpha(float):学习率
            #     m(int)样本个数
            #     n(int)变量特征
        #output: weights(mat):权重
    feature_data=np.column_stack((feature,np.ones(np.shape(feature)[0])))
    print(feature_data)
    m=np.shape(feature_data)[0]
    n=np.shape(feature_data)[1]
    weights=np.mat(np.ones((n,k)))
    i=0
    while i<=maxCycle:
        err=np.exp(feature_data*weights)
        if i%100==0:
            print ("\t-------iter:",i,\
            ",cost:",cost(err,label_data))
        rowsum=-err.sum(axis=1)
        rowsum=rowsum.repeat(k,axis=1)
        err=err/rowsum
        for x in range(m):
            err[x,label_data[x,0]]+=1
        weights=weights+(alpha/m)*feature_data.T*err
        i=i+1
    return weights

#%% 计算损失值函数
def cost(err,label_data):
    # input: err(mat):exp的值
        #          label_data:标签的值
    # output: sum_cost/ m(float):损失函数的值
    m=np.shape(err)[0]
    sum_cost=0.0
    for i in range(m):
        if err[i,label_data[i,0]]/np.sum(err[i,:])>0:
            sum_cost -= np.log(err[i,label_data[i,0]]/np.sum(err[i,:]))
        else:
            sum_cost -=0
    return sum_cost/m


#%% 导入数据
x = np.load('x_train.npy')[:,0]
y = np.load('x_train.npy')[:,1]
label = np.load('t_train.npy')

#%% 调整参数
feature=np.mat((x,y)).T
label=np.mat(label).T

#%% 训练模型
w = gradientAscent(feature,label,3,50000,0.2)
#%% 输出权重
print('权重为：',w)

#%% 预测样本
xx = np.load('x_test.npy')[:,0]
yy = np.load('x_test.npy')[:,1]
b=np.ones(10000)
test=np.mat((xx,yy,b)).T
re=test*w
predict=re.argmax(axis=1)
np.save("t_test.npy",predict)

plt.scatter(xx,yy, c=(np.ravel(predict)))
plt.show()
