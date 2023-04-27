import numpy
import math
import cv2
import os
import numpy
from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def block_view(A, block=(3, 3)):
    shape = (A.shape[0]// block[0], A.shape[1]// block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    print(shape)
    print(strides)
    return ast(A, shape= shape, strides= strides)

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

import numpy as np
from PIL import Image 
from scipy.signal import convolve2d
 
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))


deal_img_path = "/data/lbh/exper/MSCA/nhhaze_results/"
#deal_img_path = "EPDN_outdoor_results/"
gt_img_path = "/data/lbh/exper/Datasets/NH_hazy2/test/clean/"

deal_img_list = os.listdir(deal_img_path)
gt_img_list = os.listdir(gt_img_path)

img_dict = {}
for i,filename in enumerate(gt_img_list):
    #print(filename)
    first_name = filename.split(".",1)[0]
    img_dict[first_name] =[]
    print("我很累：",first_name)
    for filename_t in deal_img_list:
        first_name_t = filename_t.split(".",1)[0]
        print("看看这是啥：",first_name_t)
        if first_name_t == first_name:
          img_dict[first_name].append(filename_t)
 
print(len(img_dict.keys()))
i = 0
value_list = []
for key in img_dict:
    i = i + len(img_dict[key])    
    for filename in img_dict[key]:
       img1 = Image.open(gt_img_path+key+'.png')
       #img1 = cv2.imread(gt_img_path+key+'.png')
       img2 = Image.open(deal_img_path+filename)
       #img2 = cv2.imread(deal_img_path+filename)
       print(gt_img_path+key+'.png')
       print(deal_img_path+filename)
       if img1.size[0] - img2.size[0] != 0 or img1.size[1] - img2.size[1]:
           img2 = img2.resize((img1.size[0],img1.size[1]))
       value =  compute_ssim(np.array(img1.convert('L')), np.array(img2.convert('L')))
       #value = calculate_ssim(img1, img2)
       value_list.append(value)
       print(value)
    print("-------------------------------------------------------")
sum_value = 0
for i in value_list:
    sum_value = sum_value + i
print(sum_value)
print(len(value_list))
print(sum_value/len(value_list))
