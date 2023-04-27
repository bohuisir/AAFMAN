import torch
import torch.nn as nn
import torch.nn.functional as F
# from dcn import  DeformableConv2d
import math
from pytorch_msssim import msssim


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class MSCALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(MSCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3, dilation=3)

   
        t = int(abs((math.log(channel,2) + b)/gamma))
        k_size = t if t % 2 else t+1
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        y1 = self.avg_pool(x1)
        y2 = self.avg_pool(x2)
        y3 = self.avg_pool(x3)
      
        y1_atten = self.conv_1(y1.squeeze(-1).transpose(-1, -2))
        y1_atten = y1_atten.permute(0,2,1).unsqueeze(-1)
        y2_atten = self.conv_2(y2.squeeze(-1).transpose(-1, -2))
        y2_atten = y2_atten.permute(0, 2, 1).unsqueeze(-1)
        y3_atten = self.conv_3(y3.squeeze(-1).transpose(-1, -2))
        y3_atten = y3_atten.permute(0, 2, 1).unsqueeze(-1)

        y1_atten_score = self.soft(y1_atten)
        y2_atten_score = self.soft(y2_atten)
        y3_atten_score = self.soft(y3_atten)
        
        return x1 * y1_atten_score + x2 * y2_atten_score + x3 * y3_atten_score


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    # 初始化调用block （1024，reflect，nn.ReLU(True)，instance_norm）
    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

  
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y




class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', n_blocks=6):
        super(Base_Model,self).__init__()

        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(True))

        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))

        self.down3 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf * 4),
                                   nn.ReLU(True))

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)
        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * 4, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res = nn.Sequential(*model_res)

   
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf*2),
                                 nn.ReLU(True))


        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())



        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(448, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
         
        )
       
        self.fuse_conv1 = nn.Sequential(nn.Conv2d(448, 64, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(True))
        self.fuse_conv2 = nn.Sequential(nn.Conv2d(448, 128, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True))
        self.fuse_conv3 = nn.Sequential(nn.Conv2d(448, 256,  kernel_size=5, stride=4, padding=1),
                                        nn.InstanceNorm2d(ngf),
                                        nn.ReLU(True))

        self.pa1 = PALayer(256)
        self.pa2 = PALayer(128)
        self.pa3 = PALayer(64)


        self.ca1 = MSCALayer(256)
        self.ca2 = MSCALayer(128)
        self.ca3 = MSCALayer(64)


    def forward(self, input):


        x_down1 = self.down1(input)  # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)  # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)  # [bs, 256, 64, 64]


        x_avg1 = self.avg_pool(x_down1)
        x_avg2 = self.avg_pool(x_down2)
        x_avg3 = self.avg_pool(x_down3)
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        x_down1_reweight = x_down1 * w1
        x_down2_reweight = x_down2 * w2
        x_down3_reweight = x_down3 * w3


        fuse1 = x_down1_reweight
        fuse2 = F.interpolate(x_down2_reweight, scale_factor=2)
        fuse3 = F.interpolate(x_down3_reweight, scale_factor=4)

        fuse_feature = torch.cat((fuse1,fuse2,fuse3),dim=1)


        fuse_1 = self.fuse_conv1(fuse_feature)
        fuse_2 = self.fuse_conv2(fuse_feature)
        fuse_3 = self.fuse_conv3(fuse_feature)


        x6 = self.model_res(x_down3)
        x6 = self.ca1(x6)
        x6 = self.pa1(x6)

        x_up1 = self.up1(x6 + fuse_3)
        x_up1 = self.ca2(x_up1)
        x_up1 = self.pa2(x_up1)

        x_up2 = self.up2(x_up1 + fuse_2)
        x_up2 = self.ca3(x_up2)
        x_up2 = self.pa3(x_up2)

        x_up3 = self.up3(x_up2 + fuse_1)


        return x_up3



# 判别器
class Discriminator(nn.Module):
    def __init__(self, bn=False):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.InstanceNorm2d(256) if not bn else nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(256) if not bn else nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512) if bn else nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(512) if bn else nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)  # ,

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
