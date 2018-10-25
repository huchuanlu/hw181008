import tensorflow as tf
import numpy as np


#自定义网络
class Result(object):

    def __init__(self, results, y_train):
        super(Result, self).__init__()
        self.results = results
        self.y_train = y_train

    def iou(self):
        area_add = np.zeros((self.results.shape[0], 1))
        area_sub = np.zeros((self.results.shape[0], 1))
        area_bool = np.zeros((self.results.shape[0], 1))

        # 计算面积
        for i in range(self.y_train.shape[0]):
            area_real = (self.y_train[i][2] - self.y_train[i][0]) * (self.y_train[i][3] - self.y_train[i][1])
            area_pred = (self.results[i][2] - self.results[i][0]) * (self.results[i][3] - self.results[i][1])
            # 求交集
            # xmin
            xmin_sub = self.y_train[i][0] if self.y_train[i][0] > self.results[i][0] else self.results[i][0]
            # ymin
            ymin_sub = self.y_train[i][1] if self.y_train[i][1] > self.results[i][1] else self.results[i][1]
            # xmax
            xmax_sub = self.y_train[i][2] if self.y_train[i][2] < self.results[i][2] else self.results[i][2]
            # ymax
            ymax_sub = self.y_train[i][3] if self.y_train[i][3] < self.results[i][2] else self.results[i][3]

            # area_sub
            area_sub[i] = (xmax_sub - xmin_sub) * (ymax_sub - ymin_sub)
            # area_add
            area_add[i] = area_real + area_pred - area_sub[i]

        # area_bool
        area_bool = area_sub / area_add
        acc = np.sum(area_bool > 0.5)

        iou=acc / area_bool.shape[0]

        return iou




