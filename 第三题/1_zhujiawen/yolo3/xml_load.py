# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:36:44 2018
@author: gg
"""
#功能：读取xml文件 并转化为txt文件
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os

import os
path='E:/research/deep learning/luhuchuan_test/181008/3/trainval/box'
label=[]
def listdir(path, list_name):  #传入存储的list
   for file in os.listdir(path):
      file_path = os.path.join(path, file)
      if os.path.isdir(file_path):
         listdir(file_path, list_name)#继续深入其中的文件夹
      else:
         list_name.append(file)#保存该文件夹下所有的文件名
listdir(path,label)
#loc=[]
list_file = open('model.txt' , 'w')
for i in  range(len(label)):
    #label[i] 文件名  path_xml 路径+文件名
    path_xml=path+'/'+label[i]
    #parse函数可以解析xml文档
    ann = ET.parse(path_xml)
    #读出位置坐标
    box = [int(ann.find(tag).text)
           for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
    print(box[0])
    img_name = os.path.splitext(label[i])[0]  # 图片名字 不含后缀

    writer_txt='./trainval/'+img_name+'.jpg'+' '+str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+','+'0'
    list_file.write(writer_txt)
    list_file.write('\n')
list_file.close()







'''
#txt 保存地址
save_dir = 'D:\plate_train'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
f = open(os.path.join(save_dir, 'landmark.txt'), 'w')



ann = ET.parse('E:/research/deep learning/luhuchuan_test/181008/3/trainval/box_test/water-blue-fog-river_1417.xml')
box = [int(ann.find(tag).text)-1
           for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
f.write(str(box[1]) + ' ')
f.write('\t\n')
f.close()


DOMTree = xml.dom.minidom.parse('E:/research/deep learning/luhuchuan_test/181008/3/trainval/box_test/water-blue-fog-river_1417.xml')
annotation = DOMTree.documentElement

filename = annotation.getElementsByTagName("filename")[0]
imgname = filename.childNodes[0].data + '.jpg'
print(imgname)

objects = annotation.getElementsByTagName("object")

loc = [imgname]  # 文档保存格式：文件名 坐标

for object in objects:
    bbox = object.getElementsByTagName("bndbox")[0]
    leftTopx = bbox.getElementsByTagName("leftTopx")[0]
    lefttopx = leftTopx.childNodes[0].data
    print(lefttopx)
    leftTopy = bbox.getElementsByTagName("leftTopy")[0]
    lefttopy = leftTopy.childNodes[0].data
    print(lefttopy)
    rightTopx = bbox.getElementsByTagName("rightTopx")[0]
    righttopx = rightTopx.childNodes[0].data
    print(righttopx)
    rightTopy = bbox.getElementsByTagName("rightTopy")[0]
    righttopy = rightTopy.childNodes[0].data
    print(righttopy)
    rightBottomx = bbox.getElementsByTagName("rightBottomx")[0]
    rightbottomx = rightBottomx.childNodes[0].data
    print(rightbottomx)
    rightBottomy = bbox.getElementsByTagName("rightBottomy")[0]
    rightbottomy = rightBottomy.childNodes[0].data
    print(rightbottomy)
    leftBottomx = bbox.getElementsByTagName("leftBottomx")[0]
    leftbottomx = leftBottomx.childNodes[0].data
    print(leftbottomx)
    leftBottomy = bbox.getElementsByTagName("leftBottomy")[0]
    leftbottomy = leftBottomy.childNodes[0].data
    print(leftbottomy)

    loc = loc + [lefttopx, lefttopy, righttopx, righttopy, rightbottomx, rightbottomy, leftbottomx, leftbottomy]

for i in range(len(loc)):
    f.write(str(loc[i]) + ' ')
f.write('\t\n')
f.close()
'''
