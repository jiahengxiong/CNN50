#__author__ = 'xiongjiaheng'
# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import os
from PIL import Image
import matplotlib.image as mpimg
import scipy



def IsValidImage(img_path):
    """
    判断文件是否为有效（完整）的图片
    :param img_path:图片路径
    :return:True：有效 False：无效
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
    """
    if IsValidImage(img_path):

        str = img_path.rsplit(".", 1)
        output_img_path = str[0] + ".jpg"
        print(output_img_path)
        im = Image.open(img_path)
        im=im.convert('RGB')
        im.save(output_img_path)
        return output_img_path

    else:
        return False





def predict(img_path):
    my_image = image.load_img(img_path, target_size=(32, 32))

    my_image = image.img_to_array(my_image)
    print("my_image.shape = " + str(my_image.shape))
    my_image = np.expand_dims(my_image,axis=0)/255



    print("my_image.shape = " + str(my_image.shape))

    print("class prediction vector  = ")
    model = load_model('ResNet50.h5')

    result=model.predict(my_image)
    classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    index=np.argmax(result,1)
    print(index)
    result=classes[int(index)]
    print("输入图像为：",classes[int(index)])


    plot_model(model, to_file='model.png')
    return result