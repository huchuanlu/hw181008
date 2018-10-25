"""
Class definition of YOLO_v3 style detection model on image and video
使用yolo3对图片、视频进行分类
"""
import colorsys
import os
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
from yolo import YOLO
from xml.etree import ElementTree as ET


# 测试我的图片数据集
#建立模型
mymodel=YOLO(model_path='model_data/trained_weights_final.h5', score=0.0,
        iou = 0.0)

annotation_path='label/test.txt'
box_testpath = './test/box/'

with open(annotation_path) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].split()#按空格区分开  图片路径+名字
        image = Image.open(line[0])#读取图片
        w, h = image.size
        topp, leftt, bottomm, rightt = mymodel.detect_image(image)
        print('picture',i)

        pre_xmin = leftt
        pre_ymin = bottomm
        pre_xmax = rightt
        pre_ymax = topp

        annotation = ET.Element("prediction")
        bnbox = ET.SubElement(annotation, "bndbox")
        xmin = ET.SubElement(bnbox, "xmin")
        ymin = ET.SubElement(bnbox, "ymin")
        xmax = ET.SubElement(bnbox, "xmax")
        ymax = ET.SubElement(bnbox, "ymax")
        conf = ET.SubElement(bnbox, "conf")

        # 限幅
        if pre_xmax > w:
            det_xmax = w
        if pre_ymax > h:
            pre_ymax = h
        if pre_xmin < 0:
            pre_xmin = 0
        if pre_ymin < 0:
            pre_ymin = 0

        xmin.text = str(int(round(pre_xmin)))
        ymin.text = str(int(round(pre_ymin)))
        xmax.text = str(int(round(pre_xmax)))
        ymax.text = str(int(round(pre_ymax)))
        conf.text = str(float(1))

        tree = ET.ElementTree(annotation)
        tree.write(box_testpath + line[0].split('/',-1)[-1][:-3] + 'xml')

mymodel.close_session()


