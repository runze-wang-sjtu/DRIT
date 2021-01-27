# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2021/1/27 10:16 上午
import torch.nn as nn

def dice_loss(input, target, n_classes=4, class_weights=None):
    smooth = 1.
    loss = 0.
    m = nn.Softmax(dim=1)
    input = m(input)
    for c in range(n_classes):
        iflat = input[:, c].view(-1)
        tflat = target[:, c].view(-1)
        intersection = (iflat * tflat).sum()

        if class_weights is not None:
            w = class_weights[c]
            loss += w * (1 - ((2. * intersection + smooth) /
                              (iflat.sum() + tflat.sum() + smooth)))
        else:
            loss += 1 - ((2. * intersection + smooth) /
                         (iflat.sum() + tflat.sum() + smooth))
    return loss
