import torch
import time
import argparse
from Model import Base_Model, Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr, to_ssim_skimage
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
import math
import cv2


# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='model_test')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='/data/lbh/exper/Datasets/Dataset/dataset_thin/test')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--model', default='',type=str,help='load test model')
parser.add_argument('--padding', type=bool, default=False)
args = parser.parse_args()

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top,padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top,padding_top + image.size(2)


val_dataset = os.path.join(args.data_dir, 'hazy')
predict_result = args.predict_result
test_batch_size = args.test_batch_size
test_model = args.model
test_padding = args.padding

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join(args.model_save_dir, '')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = Base_Model(3, 3)
print('model parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))

val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)

# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load(test_model))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    time_list = []
    MyEnsembleNet.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for batch_idx, (hazy, name) in enumerate(val_loader):
        # print(len(val_loader))
        start = time.time()
        hazy = hazy.to(device)
        name = "".join(name)

        h, w = hazy.shape[2], hazy.shape[3]

        if test_padding==True:
            max_h = int(math.ceil(h / 4)) * 4
            max_w = int(math.ceil(w / 4)) * 4
            hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

            img_tensor = MyEnsembleNet(hazy)
            img_tensor = img_tensor.data[:, :, ori_top:ori_down, ori_left:ori_right]
        else:
            img_tensor = MyEnsembleNet(hazy)

        end = time.time()
        time_list.append((end - start))
        img_list.append(img_tensor)

        imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(name) + '.png'))

    time_cost = float(sum(time_list) / len(time_list))
    print('running time per image: ', time_cost)

# writer.close()








