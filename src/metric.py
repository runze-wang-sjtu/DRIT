# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2021/1/22 10:30 上午

import numpy as np


class Metrics(object):

    def __init__(self):

        self.update_num = 0
        self.metric = {'dice_0': 0, 'dice_1': 0, 'dice_2': 0, 'dice_3': 0, 'dice': 0}

    def dice_coef(self, true, pred):

        true_f = true.flatten()
        pred_f = pred.flatten()
        intersection = np.sum(true_f * pred_f)
        smooth = 0.0001

        return (2. * intersection + smooth) / (np.sum(true_f) + np.sum(pred_f) + smooth)

    def dice_coef_multilabel(self, true, pred, numlabels):

        dice = 0
        for index in range(numlabels):
            if index == 0:
                self.dice_0 = self.dice_coef(true[:, index, ::], pred[:, index, ::])
            elif index == 1:
                self.dice_1 = self.dice_coef(true[:, index, ::], pred[:, index, ::])
            elif index == 2:
                self.dice_2 = self.dice_coef(true[:, index, ::], pred[:, index, ::])
            elif index == 3:
                self.dice_3 = self.dice_coef(true[:, index, ::], pred[:, index, ::])
            dice += self.dice_coef(true[:, index, ::], pred[:, index, ::])

        self.dice_multilabel = dice / numlabels

        return self.dice_0, self.dice_1, self.dice_2, self.dice_3, self.dice_multilabel

    def update(self):

        self.metric['dice_0'] += self.dice_0
        self.metric['dice_1'] += self.dice_1
        self.metric['dice_2'] += self.dice_2
        self.metric['dice_3'] += self.dice_3
        self.metric['dice'] += self.dice_multilabel
        self.update_num += 1

    def mean(self):

        self.metric['dice_0'] = self.metric['dice_0'] / self.update_num
        self.metric['dice_1'] = self.metric['dice_1'] / self.update_num
        self.metric['dice_2'] = self.metric['dice_2'] / self.update_num
        self.metric['dice_3'] = self.metric['dice_3'] / self.update_num
        self.metric['dice'] = self.metric['dice'] / self.update_num

        return self.metric
