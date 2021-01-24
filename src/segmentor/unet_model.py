#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   unet_model.py
@Time    :   2021/01/18 19:08:13
@Author  :   Runze Wang
@Contact :   runze.wang@sjtu.edu.cn
'''

# here put the import lib
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_A = DoubleConv(n_channels, 64)
        self.down1_A = Down(64, 128)
        self.down2_A = Down(128, 256)
        self.down3_A = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4_A = Down(512, 1024 // factor)
        self.up1_A = Up(1024, 512 // factor, bilinear)
        self.up2_A = Up(512, 256 // factor, bilinear)
        self.up3_A = Up(256, 128 // factor, bilinear)
        self.up4_A = Up(128, 64, bilinear)
        self.outc_A = OutConv(64, n_classes)

        self.inc_B = DoubleConv(n_channels, 64)
        self.down1_B = Down(64, 128)
        self.down2_B = Down(128, 256)
        self.down3_B = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4_B = Down(512, 1024 // factor)
        self.up1_B = Up(1024, 512 // factor, bilinear)
        self.up2_B = Up(512, 256 // factor, bilinear)
        self.up3_B = Up(256, 128 // factor, bilinear)
        self.up4_B = Up(128, 64, bilinear)
        self.outc_B = OutConv(64, n_classes)

    def forward_a(self, x):
        x1 = self.inc_A(x)
        x2 = self.down1_A(x1)
        x3 = self.down2_A(x2)
        x4 = self.down3_A(x3)
        x5 = self.down4_A(x4)
        x = self.up1_A(x5, x4)
        x = self.up2_A(x, x3)
        x = self.up3_A(x, x2)
        x = self.up4_A(x, x1)
        logits_A = self.outc_A(x)
        return logits_A

    def forward_b(self, x):
        x1 = self.inc_B(x)
        x2 = self.down1_B(x1)
        x3 = self.down2_B(x2)
        x4 = self.down3_B(x3)
        x5 = self.down4_B(x4)
        x = self.up1_B(x5, x4)
        x = self.up2_B(x, x3)
        x = self.up3_B(x, x2)
        x = self.up4_B(x, x1)
        logits_B = self.outc_B(x)
        return logits_B