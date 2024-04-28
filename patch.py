import glob
from pathlib import Path
from pickletools import float8
import re
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#画外边届框
def draw_min_rect_rectangle(mask_path,imgsize):
    image = cv2.imread(mask_path)
    image = cv2.resize(image,(imgsize,imgsize))
    thresh = cv2.Canny(image, 128, 256)
    global loc1
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.copy(image)
    x,y = imgsize,imgsize
    x_r,y_r = 0,0
    for cnt in contours:
        x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(cnt)
        xr_tmp = x_tmp + w_tmp
        yr_tmp = y_tmp + h_tmp
        x = min(x,x_tmp)
        y = min(y,y_tmp)
        x_r = max(x_r,xr_tmp)
        y_r = max(y_r,yr_tmp)
        # sq = w * h
        w = x_r-x
        h = y_r-y
        if w%2 != 0:
            w+=1
        if h%2 != 0:
            h+=1
        loc1 = np.array((x,y,w,h))
        # if(sq > minsq):
        #     loc = np.array((x,y,w,h))
        # 绘制矩形
        cv2.rectangle(img,  (x, y+h), (x+w, y), (255, 255, 255))
    
    return img,loc1

#填充正方形patch
    # 将最小内接矩形填充为白色
def white_rectangle(mask_path,loc,imgsize):
    image = cv2.imread(mask_path)
    image = cv2.resize(image,(imgsize,imgsize))
    x,y,w,h = loc[0],loc[1],loc[2],loc[3]
    white = [255, 255, 255]
    for col in range(x, x+w):
        for row in range(y, y+h):
            image[row, col] = white

    
    
    return image

def get_patch(mask_path,loc,imgsize):
    pic = cv2.imread(mask_path) # 读取mask对应的图片
    pic = cv2.resize(pic,(imgsize,imgsize))
    x,y,w,h = loc[0],loc[1],loc[2],loc[3]
  
    cut = pic[y:y+h, x:x+w]   # 根据内接矩形的顶点切割图片
    patch = cut[:, :, ::-1]
    patch = patch.transpose((2, 0, 1))
    patch = np.ascontiguousarray(patch)
    patch = torch.from_numpy(patch).float().unsqueeze(0)
    patch = patch / 255.0 
    return patch
    # cv2.imwrite(save_path, cut) # 保存切割好的图片


class PatchApplier(nn.Module):    
    '''
        仿射变换
        1、得到被攻击图片的shape
        2、以patch为中心 加边到攻击图片尺寸 上下：(整高-patch高)/2 左右：(整宽- patch宽)/2
        3、nnConstantPad2d
        4、affine_grid
    '''
    def __init__(self):
        super(PatchApplier, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.minangle = -5/180*math.pi
        self.maxangle = 5/180*math.pi

    def forward(self,clear_image,patch,mask,loc,img_size,trans=False):
        batch_size = 1
        #变换
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, patch.size(-3), patch.size(-2), patch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size())) 

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1,patch.size(-3), patch.size(-2), patch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) 
        # # Create random noise tensor
        # noise = torch.cuda.FloatTensor(patch.size()).uniform_(-1, 1) * noise_factor
        if(trans):
            patch = patch * contrast + brightness


        padw = (img_size - loc[2])/2
        padh = (img_size-loc[3])/2
        mypad = nn.ConstantPad2d((int(padw),int(padw),int(padh),int(padh)),0)
        patch_mask = mypad(patch).to(device)
        mask_img = mypad(mask).to(device)
        #变换矩阵
        target_size = loc[2]*loc[3]
        current_size = (img_size - int(padw)*2) * (img_size - int(padh)*2)
        current_size = torch.tensor(current_size).to(device)
        scale = target_size / current_size
        if trans:
            angle = torch.cuda.FloatTensor(batch_size).uniform_(self.minangle, self.maxangle)    
        else:
            angle = torch.cuda.FloatTensor(1).fill_(0) #1 batchsize
               

        #目标中心点相对于当前中心点位置(左上角)
        #targt：loc[0],loc[1] now:(416-w)/2 ，(416-h)/2 ->padw.padh
        tx = (padw - loc[0])/208
        ty = (padh - loc[1])/208
        # print(padw,padh)
        sin = torch.sin(angle)
        cos = torch.cos(angle)        

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(1 ,2, 3).fill_(0)  # torch.Size([ 2, 3])
        theta[:, 0, 0] = (cos/scale)
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = (tx*cos/scale+ty*sin/scale)
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = (cos/scale)
        theta[:, 1, 2] = (-tx*sin/scale+ty*cos/scale)


        grid = F.affine_grid(theta,clear_image.size())  # torch.Size([1，416, 416, 2])

        patch_mask = F.grid_sample(patch_mask, grid)
        mask_img = F.grid_sample(mask_img, grid)
        # patch_mask = torch.where((mask_img == 0), mask_img, patch_mask)

        # adv_image = torch.where((patch_mask == 0), clear_image, patch_mask)
        adv_image = mask_img * patch_mask + (1-mask_img) * clear_image
        return adv_image

class PhyPatchApplier(nn.Module):    
    '''
        仿射变换
        1、得到被攻击图片的shape
        2、以patch为中心 加边到攻击图片尺寸 上下：(整高-patch高)/2 左右：(整宽- patch宽)/2
        3、nnConstantPad2d
        4、affine_grid
    '''
    def __init__(self):
        super(PhyPatchApplier, self).__init__()
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.3
        self.max_brightness = 0.2
        # self.noise_factor = 0.1
        self.minangle = -10/180*math.pi
        self.maxangle = 10/180*math.pi
        self.maxscale = 0.5
        self.minscale = 0.2
        self.maxmove = 0
        self.minmove = -0

    def forward(self,clear_image,patch,mask_img):
        batch_size = 1
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, patch.size(-3), patch.size(-2), patch.size(-1))
        contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size())) 

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1,patch.size(-3), patch.size(-2), patch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) 
        # # Create random noise tensor
        # noise = torch.cuda.FloatTensor(patch.size()).uniform_(-1, 1) * self.noise_factor

        patch = patch * contrast + brightness

        scale = torch.cuda.FloatTensor(batch_size).uniform_(self.minscale,self.maxscale)
        tx = torch.cuda.FloatTensor(batch_size).uniform_(self.minmove,self.maxmove)
        ty = torch.cuda.FloatTensor(batch_size).uniform_(self.minmove,self.maxmove)
        angle = torch.cuda.FloatTensor(batch_size).uniform_(self.minangle, self.maxangle)

       
        #变换矩阵
       
        sin = torch.sin(angle)
        cos = torch.cos(angle)        

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(1 ,2, 3).fill_(0)  # torch.Size([ 2, 3])
        theta[:, 0, 0] = (cos/scale)
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = (tx*cos/scale+ty*sin/scale)
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = (cos/scale)
        theta[:, 1, 2] = (-tx*sin/scale+ty*cos/scale)
    

        grid = F.affine_grid(theta,clear_image.size())  # torch.Size([1，416, 416, 2])

        patch = F.grid_sample(patch, grid)
        mask_img = F.grid_sample(mask_img, grid)
        adv_image = mask_img * patch + (1-mask_img) * clear_image
        return adv_image

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


def patchApplier1(clear_image,patch,mask_img,loc,img_size): 
    '''
        直接加边
        1、得到被攻击图片的shape(w,h)
        2、上下左右要加的像素值--上/左：图片坐标，下/右：(h-上-patch高)/(w-左-patch宽)
        3、nnConstantPad2d
    '''

    padt = loc[1]
    padb = img_size - loc[1] - loc[3]
    padl = loc[0]
    padr = img_size - loc[0] - loc[2]
    mypad = nn.ConstantPad2d((padl,padr,padt,padb),0)
    patch_mask = mypad(patch).to(device)
    patch_mask = patch_mask * mask_img

    adv_image = torch.where((patch_mask != 1), clear_image, patch_mask)

    return adv_image