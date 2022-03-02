# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:28:00 2021

@author: MaYiming
"""
#导入库
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import numpy as np
#训练集类
class mydata(Dataset):
    #传入一个params字典
    def __init__(self,params):
        super(mydata,self).__init__()                       #继承
        self.train_image_path = params["train_image_path"]  #训练集图片路径
        self.train_label_path = params["train_label_path"]  #训练集标签路径
        #由于labels在本次实验是以txt存储的，所以使用下边的方式直接读取存入列表
        self.train_labels = open(self.train_label_path,'r').read().splitlines()
        self.train_num = len(self.train_labels)             #训练集数量
        self.length = 28                                    #对图片预处理的长宽大小
    #长度
    def __len__(self):
        return self.train_num
    #获得元素（传入的参数是图片的下标）
    def __getitem__(self,index):
        #图片路径
        img_path = os.path.join(self.train_image_path,str(index)+'.png')
        #读取图片并利用transform函数转化为tensor
        img = self.transform(Image.open(img_path))
        #该图片对应标签
        lb = int(self.train_labels[index])
        return img,lb
    
    def transform(self,img):
        # 随机初始化
        #以一定的概率反转图片
        p = np.random.rand(1)
        if p >= 0.8:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 旋转图片
        angle = np.random.uniform(-15, 15)
        img.rotate(angle)
        #对图片进行统一化
        trans_way = transforms.Compose([transforms.Resize((self.length,self.length)),
                                        transforms.ToTensor()])
        trans_img = trans_way(img)
        return trans_img

#测试数据类操作同上
class myTestdata(Dataset):
    def __init__(self,params):
        super(myTestdata,self).__init__()
        self.test_image_path = params["test_image_path"]
        self.test_label_path = params["test_label_path"]
        self.test_labels = open(self.test_label_path,'r').read().splitlines()
        self.test_num = len(self.test_labels)
        self.length = 28
        
    def __len__(self):
        return self.test_num
    
    def __getitem__(self,index):
        img_path = os.path.join(self.test_image_path,str(index)+'.png')
        img = self.transform(Image.open(img_path))
        lb = int(self.test_labels[index])
        return img,lb
    
    def transform(self,img):
        # flip, crop, rotate
        p = np.random.rand(1)
        if p >= 0.7:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # rotate
        angle = np.random.uniform(-18, 18)
        img.rotate(angle)
        
        trans_way = transforms.Compose([transforms.Resize((self.length,self.length)),
                                        transforms.ToTensor()])
        trans_img = trans_way(img)
        return trans_img