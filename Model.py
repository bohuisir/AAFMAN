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

        # 一维卷积的定义
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
        # 全局平均池化
        y1 = self.avg_pool(x1)
        y2 = self.avg_pool(x2)
        y3 = self.avg_pool(x3)
        # 1D卷积过程
        y1_atten = self.conv_1(y1.squeeze(-1).transpose(-1, -2))
        y1_atten = y1_atten.permute(0,2,1).unsqueeze(-1)
        y2_atten = self.conv_2(y2.squeeze(-1).transpose(-1, -2))
        y2_atten = y2_atten.permute(0, 2, 1).unsqueeze(-1)
        y3_atten = self.conv_3(y3.squeeze(-1).transpose(-1, -2))
        y3_atten = y3_atten.permute(0, 2, 1).unsqueeze(-1)

        y1_atten_score = self.soft(y1_atten)
        y2_atten_score = self.soft(y2_atten)
        y3_atten_score = self.soft(y3_atten)
        # print(y3_atten_score.shape)
        # 自适应特征融合
        # zong = x1 * y1_atten_score + x2 * y2_atten_score + x3 * y3_atten_score
        # print("zong:",zong.shape)
        return x1 * y1_atten_score + x2 * y2_atten_score + x3 * y3_atten_score


# class CALayer(nn.Module):
#     def __init__(self, channel):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ca(y)
#         return x * y

# class ResnetGlobalAttention(nn.Module):
#     def __init__(self, channel, gamma=2, b=1):
#         super(ResnetGlobalAttention, self).__init__()
#
#         self.feature_channel = channel
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         t = int(abs((math.log(channel, 2) + b) / gamma))
#         k_size = t if t % 2 else t + 1
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         # self.relu = nn.ReLU(inplace=True)
#         self.soft = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#
#         # 添加全局信息
#         zx = y.squeeze(-1)
#         zy = zx.permute(0, 2, 1)
#         zg = torch.matmul(zy, zx)
#
#
#         batch = zg.shape[0]
#         v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
#         v = v.unsqueeze_(-1).permute(1, 2, 0)
#
#         # 全局局部信息融合
#         atten = self.conv(y.squeeze(-1).transpose(-1, -2))
#         atten = atten + v
#         # atten = self.relu(atten)
#         atten = self.conv_end(atten)
#         atten = atten.permute(0,2,1).unsqueeze(-1)
#
#         atten_score = self.soft(atten)
#
#         return x * atten_score


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

        # 通道不变卷积  padding=0， 尺寸不变
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        # 这个没有用到
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
        # 图像经过两次卷积并且在加上原图 ，更换FFA在这个模块
        out = x + self.conv_block(x)
        return out


# class MFF(nn.Module):
#
#     def __init__(self, feature_channel=64, gamma=2, b=1):
#         super(MFF, self).__init__()
#         if feature_channel == 128:
#             self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
#                                        nn.ReLU(inplace=False))
#             self.conv2 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
#                                        nn.ReLU(inplace=False))
#             self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel*2, feature_channel, kernel_size=3, stride=2, padding=1,output_padding=1),
#                                        nn.ReLU(inplace=False))
#
#             self.feature_channel = feature_channel
#
#
#         elif feature_channel == 64:
#             self.conv1 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
#                                        nn.ReLU(inplace=False))
#             self.conv2 = nn.Sequential(nn.ConvTranspose2d(feature_channel * 2, feature_channel, kernel_size=3, stride=2, padding=1,output_padding=1),
#                                        nn.ReLU(inplace=False))
#             self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel * 4, feature_channel, kernel_size=5, stride=4, padding=1,output_padding=1),
#                                        nn.ReLU(inplace=False))
#
#             self.feature_channel = feature_channel
#
#         else:
#             self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 4, feature_channel, kernel_size=5, stride=4, padding=1),
#                                        nn.ReLU(inplace=False))
#             self.conv2 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel, kernel_size=3, stride=2, padding=1),
#                                        nn.ReLU(inplace=False))
#             self.conv3 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel, kernel_size=1, stride=1, padding=0),
#                                        nn.ReLU(inplace=False))
#
#             self.feature_channel = feature_channel
#
#
#         t = int(abs((math.log(feature_channel, 2) + b) / gamma))
#         k_size = t if t % 2 else t + 1
#         self.con_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.con_1_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         # self.relu1 = nn.ReLU(inplace=True)
#
#         self.con_2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.con_2_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         # self.relu2 = nn.ReLU(inplace=True)
#
#         self.con_3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.con_3_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         # self.relu3 = nn.ReLU(inplace=True)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#
#
#         # 激活函数
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, f1, f2, f3):
#
#         feature1 = self.conv1(f1).unsqueeze_(dim=1)
#         # print("f2的尺寸", f2.shape)
#         feature2 = self.conv2(f2).unsqueeze_(dim=1)
#         feature3 = self.conv3(f3).unsqueeze_(dim=1)
#
#         feature12 = torch.cat([feature1, feature2],dim=1)
#         feature123 = torch.cat([feature12, feature3],dim=1)
#
#         # 查看下尺寸
#         # print(feature123.shape)
#         # 通道相加
#         fea_U = torch.sum(feature123, dim=1)
#         # print(fea_U.shape)
#         # 相当于平均池化
#         u = self.avg_pool(fea_U)
#
#         zx = u.squeeze(-1)
#
#         zy = zx.permute(0,2,1)
#         zg = torch.matmul(zy, zx)
#         batch = zg.shape[0]
#         v = zg.squeeze(-1).permute(1,0).expand((self.feature_channel,batch))
#         v = v.unsqueeze_(-1).permute(1,2,0)
#
#         # print("v:",v.shape)
#         # 全局信息
#
#
#         # 全局局部信息融合
#         vector1 = self.con_1(u.squeeze(-1).transpose(-1, -2)) + v
#         # vector1 = self.relu1(vector1)
#         vector1 = self.con_1_end(vector1)
#
#         vector2 = self.con_2(u.squeeze(-1).transpose(-1, -2)) + v
#         # vector2 = self.relu2(vector2)
#         vector2 = self.con_2_end(vector2)
#
#         vector3 = self.con_3(u.squeeze(-1).transpose(-1, -2)) + v
#         # vector3 = self.relu3(vector3)
#         vector3 = self.con_3_end(vector3)
#
#
#         # fea_s = fea_U.mean(-1).mean(-1)
#         # # (12,64)  (12,128)
#         # # 在经过全连接层降低参数,全连接之后在经过升维的
#         # fea_z = self.fc(fea_s)
#         # # 经过降维在升维度，这部分可优化
#         # vector1 = self.fc1(fea_z).unsqueeze_(dim=1)
#         # vector2 = self.fc2(fea_z).unsqueeze_(dim=1)
#         # vector3 = self.fc3(fea_z).unsqueeze_(dim=1)
#
#         vector12 = torch.cat([vector1, vector2], dim=1)
#         vector123 = torch.cat([vector12, vector3], dim=1)
#         # 这样参数两会不会过多
#         attention_vectors = self.softmax(vector123)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feature123 * attention_vectors).sum(dim=1)
#
#         return fea_v

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


# class DehazeBlock(nn.Module):
#     def __init__(self, conv, dim, kernel_size, ):
#         super(DehazeBlock, self).__init__()
#         self.conv1 = conv(dim, dim, kernel_size, bias=True)
#         self.act1 = nn.ReLU(inplace=True)
#         self.conv2 = conv(dim, dim, kernel_size, bias=True)
#         self.calayer = CALayer(dim)
#         self.palayer = PALayer(dim)
#
#     def forward(self, x):
#         res = self.act1(self.conv1(x))
#         res = res + x
#         res = self.conv2(res)
#         res = self.calayer(res)
#         res = self.palayer(res)
#         res += x
#         return res



class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect', n_blocks=6):
        super(Base_Model,self).__init__()

        # 下采样
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

        # 上采样
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf*2),
                                 nn.ReLU(True))


        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        # CFPN操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(448, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1, padding=0, bias=True),
            # nn.Sigmoid()
        )
        # 这里原论文采用pooling方式，我们直接利用卷积，卷积到不同尺寸以得到有效特征。
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
        # print("x_down1_reweight尺寸,",x_down1_reweight.shape)
        # print("x_down2_reweight尺寸,", x_down2_reweight.shape)
        # print("x_down3_reweight尺寸,", x_down3_reweight.shape)


        # 将特征还原回原来的尺寸
        fuse1 = x_down1_reweight
        fuse2 = F.interpolate(x_down2_reweight, scale_factor=2)
        fuse3 = F.interpolate(x_down3_reweight, scale_factor=4)

        # print("fuse1尺寸,", fuse1.shape)
        # print("fuse2尺寸,", fuse2.shape)
        # print("fuse3尺寸,", fuse3.shape)

        fuse_feature = torch.cat((fuse1,fuse2,fuse3),dim=1)
        # print("融合特征尺寸：", fuse_feature.shape)


        fuse_1 = self.fuse_conv1(fuse_feature)
        fuse_2 = self.fuse_conv2(fuse_feature)
        fuse_3 = self.fuse_conv3(fuse_feature)

        # print("fuse1尺寸a,", fuse_1.shape)
        # print("fuse2尺寸a,", fuse_2.shape)
        # print("fuse3尺寸a,", fuse_3.shape)

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

            # nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
