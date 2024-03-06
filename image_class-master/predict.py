"""
author:tslgithub
email:mymailwith163@163.com
time:2022-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net
    TSL16
"""
from __future__ import print_function
from config import config
import sys,copy,shutil
import cv2
import os,time
from keras.preprocessing.image import img_to_array
import numpy as np

import tensorflow as tf
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

from Build_model import Build_model

class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

#这块代码导致className报错？？？
        # try:
        #     className = sys.argv[1]
        # except:
        #     print("use default className")
        #     className = "recognition_one"
        #
        #
        # self.className = className
        # self.test_data_path = os.path.join(config.test_data_path,self.className)

    def classes_id(self):
        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path

    def Predict(self):

        self.className = "recognition_one"
        self.test_data_path = os.path.join(config.test_data_path, self.className)
        start = time.time()
        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))



        if(self.channles == 3):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x)),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))
        elif(self.channles == 1):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x), 0),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))

        i,j,tmp = 0,0,[]

        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        path = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/recognition_one/"
        file_list = os.listdir(path)
        index = 0
        res = -1
        for img in data_list:
            img = np.array([img_to_array(img)],dtype='float')/255.0
            pred = model.predict(img).tolist()[0]
            label = self.classes_id()[pred.index(max(pred))]
            confidence = max(pred)
            print('predict label     is: ',label)
            print('predict confidect is: ',confidence)
            if label == "0":
                count0 += 1
                res = 0
            elif label == "1":
                count1 += 1
                res = 1
            elif label == "2":
                count2 += 1
                res = 2
            elif label == "3":
                count3 += 1
                res = 3
            index = index + 1
            if label != self.className:
                print('____________________wrong label____________________', label)
                i+=1
            else:
                j+=1

        accuracy = (1.0*j/ (1.0*len(data_list)))*100.0
        print("accuracy:{:.5}%".format(str(accuracy) ))
        print('Done')
        end = time.time()
        print("usg time:",end - start)
        print("判定为0数量：",count0)
        print("判定为1数量：",count1)
        print("判定为2数量：",count2)
        print("判定为3数量：",count3)

        with open("testLog/accuacy.txt","a") as f:
            f.write(config.model_name+","+self.className+","+"{:.5}%".format(str(accuracy))+"\n")

        return str(res) + ',' + str(end - start)

def main():
    predict = PREDICT(config)
    predict.Predict()

if __name__=='__main__':
    main()
