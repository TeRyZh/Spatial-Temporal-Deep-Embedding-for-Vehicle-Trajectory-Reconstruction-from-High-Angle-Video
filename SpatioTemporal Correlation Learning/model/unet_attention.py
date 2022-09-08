""" Full assembly of the parts to form the complete network """

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import make_grid

# Data manipulations
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up_sample(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)

        print(g1.shape)

        x1 = self.W_x(skip_connection)

        print(x1.shape)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):

    def __init__(self, 
                in_channels = 3, 
                out_channels = 2,
                nfeatures = [16,32,64,128,256],
                emd=16,
                if_sigmoid=False):

        super(AttentionUNet, self).__init__()

        self.if_sigmoid = if_sigmoid

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = ConvBlock(in_channels, nfeatures[0])
        self.Conv1 = ConvBlock(nfeatures[0], nfeatures[1])
        self.Conv2 = ConvBlock(nfeatures[1], nfeatures[2])
        self.Conv3 = ConvBlock(nfeatures[2], nfeatures[3])
        self.Conv4 = ConvBlock(nfeatures[3], nfeatures[4])

        self.Up1 = UpConv(nfeatures[4], nfeatures[4])
        self.Att1 = AttentionBlock(F_g=nfeatures[3], F_l=nfeatures[3], n_coefficients = nfeatures[3])
        self.up1_conv = ConvBlock(nfeatures[4], nfeatures[3])

        self.Up2 = UpConv(nfeatures[4], nfeatures[3])
        self.Att2 = AttentionBlock(F_g=nfeatures[2], F_l=nfeatures[2], n_coefficients = nfeatures[2])
        self.up2_conv= ConvBlock(nfeatures[3], nfeatures[2])

        self.Up3 = UpConv(nfeatures[3], nfeatures[2])
        self.Att3 = AttentionBlock(F_g=nfeatures[1], F_l=nfeatures[1], n_coefficients = nfeatures[1])
        self.up3_conv = ConvBlock(nfeatures[2], nfeatures[1])

        self.Up4 = UpConv(nfeatures[2], nfeatures[1])
        self.Att4 = AttentionBlock(F_g=nfeatures[0], F_l=nfeatures[0], n_coefficients = nfeatures[0])
        self.up4_conv = ConvBlock(nfeatures[1], nfeatures[0])

        self.outConv1 = OutConv(nfeatures[4], emd)
        self.outConv2 = OutConv(nfeatures[4], emd)
        self.outConv3 = OutConv(nfeatures[3], emd)
        self.outConv4 = OutConv(nfeatures[2], emd)
        self.outconv_emb = OutConv(nfeatures[1], emd)


        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], out_channels, 1)
        )


    def forward(self, x):
        """ 
        e : encoder layers 
        d : decoder layers 
        s : skip-connections from encoder layers to decoder layers 
        """
        e1 = self.Conv0(x) # 512*512*16 

        e2 = self.MaxPool(e1) #  256*256*16 
        e2 = self.Conv1(e2) # 256*256*32 

        e3 = self.MaxPool(e2)  #  128*128*32 
        e3 = self.Conv2(e3)  # 128*128*64 

        e4 = self.MaxPool(e3)  #  64*64*64 
        e4 = self.Conv3(e4)  # 64*64*128 

        e5 = self.MaxPool(e4)  #  32*32*128 
        e5 = self.Conv4(e5)  # 32*32*256 
        x_emb1 = self.outConv1(e5)   # 32*32*16 

        d1 = self.Up1(e5)    # 64*64*256   bridge layer 
        d1 = self.up1_conv(d1)           #  64*64*128 
        # print("d1 shape: ", d1.shape) 
        # print("e4 shape: ", e4.shape) 
        s1 = self.Att1(gate=d1, skip_connection=e4)    #  64*64*128 
        d1 = torch.cat((s1, d1), dim=1)  #  64*64*(128+128)  concatenate attention-weighted skip connection with previous layer output 
        x_emb2 = self.outConv2(d1)  # 64*64*16 

        d2 = self.Up2(d1)    #  128*128*128
        d2 = self.up2_conv(d2)   #  128*128*64
        # print("d2 shape: ", d2.shape) 
        # print("e3 shape: ", e3.shape) 
        s2 = self.Att2(gate=d2, skip_connection=e3)  #  128*128*64 
        d2 = torch.cat((s2, d2), dim=1)   #  128*128*(64+64) 
        x_emb3 = self.outConv3(d2)   #  128*128*16 

        d3 = self.Up3(d2)     #  256*256*64 
        d3 = self.up3_conv(d3)          #  256*256*32
        s3 = self.Att3(gate=d3, skip_connection=e2)  #  256*256*32 
        d3 = torch.cat((s3, d3), dim=1)   #  256*256*(32+32) 
        x_emb4 = self.outConv4(d3)      #  256*256*16

        d4 = self.Up4(d3)     #  512*512*16 
        d4 = self.up4_conv(d4)       #  512*512*16 
        s4 = self.Att4(gate=d4, skip_connection=e1)  #  512*512*16 
        d4 = torch.cat((s4, d4), dim=1)     #  512*512*(16+16) 
        x_emb5 = self.outconv_emb(d4)   #  512*512*16 

        binary_seg = self.binary_seg(d4) 

        if self.if_sigmoid: 
            binary_seg = torch.sigmoid(binary_seg) 

        return x_emb1, x_emb2, x_emb3, x_emb4, x_emb5, binary_seg 



if __name__ == '__main__':
    import numpy as np
    from ptflops import get_model_complexity_info

    x = torch.Tensor(np.random.random((4, 3, 512, 512)).astype(np.float32)).cuda()
    # model = AttentionUNet(out_channels=2).cuda()
    # model = AttentionUNet().cuda()
    model = AttentionUNet(nfeatures=[64,128,256,512,1024]).cuda()  # 469.93 GMac, 44.66 M
    # model = AttentionUNet(nfeatures=[16,32,64,128,256]).cuda()  # 29.91 GMac, 2.81M


    # emb1, emb2, emb3, emb4, emb, mask = model(x)
    # # emb, mask = model(x)
    # print(emb1.shape, emb2.shape, emb3.shape, emb4.shape, emb.shape)
    # print(emb.shape)
    # print(mask.shape)

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))