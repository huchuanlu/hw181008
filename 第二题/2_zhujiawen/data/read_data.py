import os
import numpy as np
import glob

from skimage import io,transform
from PIL import Image

class Mydata(object):
    def __init__(self, img_path,img_path_tes,img_width,img_height):
        super(Mydata, self).__init__()
        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.img_path_tes = img_path_tes

    def getnpy(self):
        all_file = []
        for file_name in os.listdir(self.img_path):
            all_file.append(file_name)
        #print(len(all_file))
        x_train = np.zeros((len(all_file), self.img_width, self.img_height, 3))
        labels_train = np.zeros((len(all_file),))
        for i in range(len(all_file)):
            img = Image.open(self.img_path + all_file[i])
            labels_train[i]=ord(all_file[i][0]) - 65
            img_resize = img.resize((self.img_width, self.img_height))
            x_train[i] = np.array(img_resize)
        #print(x_train[1].shape)
        #print(labels_train)

        all_file_tes = []
        for file_name in os.listdir(self.img_path_tes):
            all_file_tes.append(file_name)
        # #print(len(all_file_tes))
        # x_test = np.zeros((len(all_file_tes), self.img_width, self.img_height, 3))
        # for i in range(len(all_file_tes)):
        #     img = Image.open(self.img_path_tes + all_file_tes[i])
        #     img_resize = img.resize((self.img_width, self.img_height))
        #     x_test[i] = np.array(img_resize)

        name_list = [x for x in os.listdir(self.img_path_tes)]
        name_list.sort(key=lambda x: int(x[:-4]))
        cate = [self.img_path_tes + item for item in name_list]
        #x_test=[]
        i=0
        x_test = np.zeros((len(all_file_tes), self.img_width, self.img_height, 3))
        for idx, folder in enumerate(cate):
            for im in glob.glob(folder):
                img = io.imread(im)
                img = transform.resize(img, (self.img_width, self.img_height))
                x_test[i]=img
                i+=1



        labels_tes=np.load('y_test.npy')

        N, D = labels_tes.shape
        y_tes = np.zeros(N * D)
        y_tes = labels_tes.T[0][:]

        #print(x_test[1].shape)
        #print(y_tes)

        #np.save("x_train.npy", x_train)
        #np.save("y_train.npy", labels_train)

        return x_train,labels_train,x_test,y_tes
