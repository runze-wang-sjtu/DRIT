# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2021/1/22 4:15 下午
import torch
from torch import nn
from torch.nn import functional as F

import segmentor.extractors

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',
                 pretrained=False):
        super().__init__()
        self.feats_A = getattr(segmentor.extractors, backend)(pretrained)
        self.psp_A = PSPModule(psp_size, 1024, sizes)
        self.drop_1_A = nn.Dropout2d(p=0.3)
        self.up_1_A = PSPUpsample(1024, 256)
        self.up_2_A = PSPUpsample(256, 64)
        self.up_3_A = PSPUpsample(64, 64)

        self.drop_2_A = nn.Dropout2d(p=0.15)
        self.final_A = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

        self.feats_B = getattr(segmentor.extractors, backend)(pretrained)
        self.psp_B = PSPModule(psp_size, 1024, sizes)
        self.drop_1_B = nn.Dropout2d(p=0.3)
        self.up_1_B = PSPUpsample(1024, 256)
        self.up_2_B = PSPUpsample(256, 64)
        self.up_3_B = PSPUpsample(64, 64)

        self.drop_2_B = nn.Dropout2d(p=0.15)
        self.final_B = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward_a(self, x):
        f, class_f = self.feats_A(x)
        p = self.psp_A(f)
        p = self.drop_1_A(p)

        p = self.up_1_A(p)
        p = self.drop_2_A(p)

        p = self.up_2_A(p)
        p = self.drop_2_A(p)

        p = self.up_3_A(p)
        p = self.drop_2_A(p)

        return self.final_A(p)

    def forward_b(self, x):
        f, class_f = self.feats_B(x)
        p = self.psp_B(f)
        p = self.drop_1_B(p)

        p = self.up_1_B(p)
        p = self.drop_2_B(p)

        p = self.up_2_B(p)
        p = self.drop_2_B(p)

        p = self.up_3_B(p)
        p = self.drop_2_B(p)

        return self.final_B(p)
