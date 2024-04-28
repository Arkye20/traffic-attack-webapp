from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import cv2
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(img_path,img_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    # 为img增加一个维度：1
    # 因为神经网络的输入为 4 维
    img = img.unsqueeze(0)
    return img


class Content_Loss(nn.Module):
    # 其中 target 表示 C ，input 表示 G，alpha 表示 weight 的平方
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        # detach 可以理解为使 target 能够动态计算梯度
        # target 表示目标内容，即想变成的内容
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        # 将特征图变换为 2 维向量
        feature = input.view(a * b, c * d)
        # 内积的计算方法其实就是特征图乘以它的逆
        gram = torch.mm(feature, feature.t())
        # 对得到的结果取平均值
        gram /= (a * b * c * d)
        return gram

class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        # weight 和内容函数相似，表示的是权重 beta
        self.weight = weight
        # targer 表示图层目标。即新图像想要拥有的风格
        # 即保存目标风格
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        # 加权计算 input 的 Gram 矩阵
        G = self.gram(input) * self.weight
        # 计算真实的风格和想要得到的风格之间的风格损失
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out
    # 向后传播

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss




    '''
    一个正向传播过程，得到：
      输出一个具体的神经网络模型，
      一个风格损失函数集合（其中包含了 5 个不同风格目标的损失函数）
      一个内容损失函数集合（这里只有一个，你也可以多定义几个）
    '''
def get_style_model_and_loss(style_img, content_img, cnn, style_weight=1000, content_weight=1):
    # 用列表来存上面6个损失函数
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    model = nn.Sequential()
    model = model.to(device)

    content_loss_list = []
    style_loss_list = []

    # 风格提取函数
    gram = Gram()
    gram = gram.to(device)

    i = 1
    # 遍历 VGG19 ，找到其中我们需要的卷积层
    for layer in cnn:
        # 如果 layer 是  nn.Conv2d 对象，则返回 True
        # 否则返回 False
        if isinstance(layer, nn.Conv2d):
            # 将该卷积层加入我们的模型中
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            # 判断该卷积层是否用于计算内容损失
            if name in content_layers:
                # 这里是把目标放入模型中，得到该层的目标
                target = model(content_img)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的内容损失
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            # 和内容损失相似，不过增加了一步：提取风格
            if name in style_layers:
                target = model(style_img)
                target = gram(target)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的风格损失
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        # 对于池化层和 Relu 层我们直接添加即可
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)
 
    return model, style_loss_list, content_loss_list

def get_input_param_optimier(input_img):
    # 将input_img的值转为神经网络中的参数类型
    input_param = nn.Parameter(input_img.data)
    # 告诉优化器，我们优化的是 input_img 而不是网络层的权重
    # 采用 LBFGS 优化器
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer