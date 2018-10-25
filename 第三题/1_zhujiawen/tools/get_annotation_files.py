#对测试图片生成annotation_txt文件

import numpy as np
import os

img_testpath = '/home/jill/研0考试/3-yolo/test/img/'

def _main():
    file = open('/home/jill/研0考试/3-yolo/label/test.txt', 'w')
    for filename in os.listdir(img_testpath):
        file.write('./test/img/'+filename+'\n')
    file.close()
    print("have write !")

if __name__ == '__main__':
    _main()