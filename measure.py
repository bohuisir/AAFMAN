from PIL import Image
# import cv2
import numpy as np
from pathlib import Path
import math
# import yaml
import json
import os
from skimage.metrics import structural_similarity as ssim

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        print(im1.shape, im2.shape)
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        print(im1.shape)
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


# file = '/home/sh/lzy/c2/myconfig/test/sots_test.yaml'
# opt = yaml.safe_load(open(file, 'r', encoding='utf-8').read())

# img1 = np.array(Image.open('original.jpg'))
# img2 = np.array(Image.open('compress.jpg'))


# def psnr(img1, img2):
#     mse = np.mean((img1 -img2 )**2)
#     if mse == 0:
#     else:
#         return 20*np.log10(255/np.sqrt(mse))
def _psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# label_dir = 'data/Densehaze_datasets/test/clear'
# out_dir = 'pred_FFA_DENSE'
# label_dir = opt['label_dir']
# out_dir = opt['out_dir']
def get_label_name(name, format, type=0):
    if type == 0:
        return name.split('/')[-1].split('_')[0] + format
    if type == 1:
        return name + format


def measure(out_dir, label_dir, is_psnr=True, is_ssim=False, flag=True, type=0):
    # s1 = set(os.path.splitext(i)[0] for i in os.listdir(out_dir) if os.path.splitext(i)[1]=='.json' )
    save_dir, save_file = os.path.dirname(out_dir), os.path.basename(out_dir)
    save_file += '.json'
    save_path = os.path.join(save_dir, save_file)
    imgs = list(Path(out_dir).iterdir())
    format = Path(label_dir).iterdir().__next__().suffix
    psnrs = []
    ssims = []
    for img in imgs:
        # label = img.stem.split('/')[-1].split('_')[0] + format
        label = get_label_name(img.stem, format, type=type)
        label = label_dir + '/' + label

        img = Image.open(str(img))
        label = Image.open(str(label))

        if is_psnr:
            psnrs.append(_psnr(np.array(img), np.array(label)))
        if is_ssim:
            if flag:
                ssims.append(ssim(np.array(img), np.array(label), multichannel=True))
            else:
                ssims.append(compute_ssim(np.array(img.convert('L')), np.array(label.convert('L'))))
    dic = {}
    if is_psnr:
        dic['eval_psnrs'] = np.mean(np.array(psnrs))

    if is_ssim:
        dic['eval_ssims'] = np.mean(np.array(ssims))

    with open(save_path, 'w') as f:
        json.dump(dic, f)


if __name__ == "__main__":
    out_dir = '/home/lzy/experiment/newbase/mytest/a/outdoor_results'
    label_dir = '/home/lzy/experiment/newbase/mytest/a/clean'
    measure(out_dir, label_dir, True, True, True, 1)

