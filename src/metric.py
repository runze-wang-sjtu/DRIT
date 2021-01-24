# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2021/1/22 10:30 上午

import numpy as np

class Metrics(object):

    def __init__(self):

        self.update_num = 0
        self.metric = {'dice': 0}

    def dice_coef(self, true, pred):

        true_f = true.flatten()
        pred_f = pred.flatten()
        intersection = np.sum(true_f * pred_f)
        smooth = 0.0001

        return (2. * intersection + smooth) / (np.sum(true_f) + np.sum(pred_f) + smooth)

    def dice_coef_multilabel(self, true, pred, numlabels):

        dice = 0
        for index in range(numlabels):
            dice += self.dice_coef(true[:, index, ::], pred[:, index, ::])

        self.dice_multilabel = dice/numlabels

        return self.dice_multilabel

    def update(self):

        self.metric['dice'] += self.dice_multilabel
        self.update_num += 1

    def mean(self):

        self.metric['dice'] = self.metric['dice'] / self.update_num

        return self.metric
