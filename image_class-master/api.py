import json
from fileinput import filename
import numpy as np
from flask import Flask, request
import sys, os, time
from config import config
# from flask_cors import CORS
import cv2
import matplotlib.pyplot as plt

import base64
from flask_cors import CORS
import fast_glcm
from PIL import Image

from flask import make_response
grand_parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(grand_parentdir)
print(sys.path)

import predict as logic_class  # 逻辑函数类
import predictAll as logic_class_all  # 逻辑函数类
import predict_multi as logic_class_multi  # 逻辑函数类

app = Flask(__name__)

@app.route('/call/predict', methods=['GET', 'POST', 'PUT', 'DELETE'])
def call_function():
    path = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/recognition_one/"

    # save_out = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/1/"  # 保存图片的文件夹名称
    # 删除上一次预测的文件
    for root, dirs, files in os.walk(path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件

    # for root, dirs, files in os.walk(save_out, topdown=False):
    #     print(root)  # 各级文件夹绝对路径
    #     print(dirs)  # root下一级文件夹名称列表，如 ['文件夹1','文件夹2']
    #     print(files)  # root下文件名列表，如 ['文件1','文件2']
    #     # 第一步：删除文件
    #     for name in files:
    #         os.remove(os.path.join(root, name))  # 删除文件

    if request.method == 'POST':
        a = request.files.get('filename')
        print(a.filename)

        file_path = path + a.filename  # 完整的保存路径加图片名
        a.save(file_path)  # 保存

        # file_list = os.listdir(file_root)

        # for img_name in file_list:
        #     img = r'F:/硕士论文/eeg-master/image_class-master/dataset/test/0/' + str(a.filename)
        #
        #     save_path = save_out + str(a.filename)
        #
        #     img = np.array(Image.open(img).convert('L'))
        #
        #     mean = fast_glcm.fast_glcm_mean(img)
        #
        #     # plt.figure(figsize=(8,4.5))
        #     fs = 15
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.imshow(mean)
        #     plt.savefig(save_path, bbox_inches='tight')
        #     # cv2.imwrite(save_path, img_norm)

        predict = logic_class.PREDICT(config)
        ret = predict.Predict()
        res = ret.split(',')

        return json.dumps({'code': 200, 'res': str(res[0]), 'time': str(res[1])})


@app.route('/call/predict_multi', methods=['GET', 'POST', 'PUT', 'DELETE'])
def call_function_multi():
    path = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/recognition_multi/"

    # save_out = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/1/"  # 保存图片的文件夹名称
    # 删除上一次预测的文件
    for root, dirs, files in os.walk(path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件

    # for root, dirs, files in os.walk(save_out, topdown=False):
    #     print(root)  # 各级文件夹绝对路径
    #     print(dirs)  # root下一级文件夹名称列表，如 ['文件夹1','文件夹2']
    #     print(files)  # root下文件名列表，如 ['文件1','文件2']
    #     # 第一步：删除文件
    #     for name in files:
    #         os.remove(os.path.join(root, name))  # 删除文件

    if request.method == 'POST':

        for file in request.files.getlist('filename'):
            file_path = path + file.filename  # 完整的保存路径加图片名
            print(file.filename)
            file.save(file_path)  # 保存

        # file_list = os.listdir(file_root)

        # for img_name in file_list:
        #     img = r'F:/硕士论文/eeg-master/image_class-master/dataset/test/0/' + str(a.filename)
        #
        #     save_path = save_out + str(a.filename)
        #
        #     img = np.array(Image.open(img).convert('L'))
        #
        #     mean = fast_glcm.fast_glcm_mean(img)
        #
        #     # plt.figure(figsize=(8,4.5))
        #     fs = 15
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.imshow(mean)
        #     plt.savefig(save_path, bbox_inches='tight')
        #     # cv2.imwrite(save_path, img_norm)
        start = time.time()
        predict = logic_class_multi.PREDICT(config)
        ret = predict.Predict("recognition_multi")
        end = time.time()

        return json.dumps({'code': 200, 'res': ret, 'time': str(end - start)})

@app.route('/call/predictAll', methods=['GET', 'POST', 'PUT', 'DELETE'])
def call_function_all():
    if request.method == 'POST':
        path =r"F:/硕士论文/eeg-master/image_class-master/dataset/test/0/"

        emo_type = request.form['type']
        print("type:"+str(emo_type))

        file_root = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/" + str(emo_type) + '/'# 当前文件夹下的所有图片

        # save_out = r"F:/硕士论文/eeg-master/image_class-master/dataset/test/1/"  # 保存图片的文件夹名称
        # 删除上一次预测的文件
        for root, dirs, files in os.walk(file_root, topdown=False):
            # 第一步：删除文件
            for name in files:
                os.remove(os.path.join(root, name))  # 删除文件

        # for root, dirs, files in os.walk(save_out, topdown=False):
        #     print(root)  # 各级文件夹绝对路径
        #     print(dirs)  # root下一级文件夹名称列表，如 ['文件夹1','文件夹2']
        #     print(files)  # root下文件名列表，如 ['文件1','文件2']
        #     # 第一步：删除文件
        #     for name in files:
        #         os.remove(os.path.join(root, name))  # 删除文件

        # file_path = path + a.filename  # 完整的保存路径加图片名
        # a.save(file_path)  # 保存

        for file in request.files.getlist('filename'):
            file_path = file_root + file.filename  # 完整的保存路径加图片名
            print(file.filename)
            file.save(file_path)  # 保存

        # glcm提取纹理特征
        # file_list = os.listdir(file_root)
        #
        # for img_name in file_list:
        #     img = r'F:/硕士论文/eeg-master/image_class-master/dataset/test/0/' + str(a.filename)
        #
        #     save_path = save_out + str(a.filename)
        #
        #     img = np.array(Image.open(img).convert('L'))
        #
        #     mean = fast_glcm.fast_glcm_mean(img)
        #
        #     fs = 15
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        #     plt.imshow(mean)
        #     plt.savefig(save_path, bbox_inches='tight')

        predict = logic_class_all.PREDICT(config)
        ret = predict.Predict(emo_type)
        res = ret.split(',')

        return json.dumps(
            {'code': 200, 'res': str(res[0]),
             'time': str(res[1]),
             'count0': str(res[2]),
             'count1': str(res[3]),
             'count2': str(res[4]),
             'count3': str(res[5])
             })


CORS(app, resources=r'/*')

app.run(host="0.0.0.0", port=5000)

