# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2021/1/22 10:15 下午
import os
import torch
import numpy as np
from options import TestOptions
from dataset import dataset_segmentation
from model import DRIT
from saver import save_imgs
from metric import Metrics
from PIL import Image

def translation(x, h, w):
    x_local = x.squeeze().cpu().numpy()
    x_Image = Image.fromarray(x_local.astype('uint8'))
    x_resize = np.array(x_Image.resize((w, h), Image.NEAREST))
    x_cuda = torch.from_numpy(x_resize).unsqueeze(0).cuda()
    return x_cuda

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset_A = dataset_segmentation(opts, setname='A')
    loader_A = torch.utils.data.DataLoader(dataset_A, batch_size=1, num_workers=opts.nThreads)
    dataset_B = dataset_segmentation(opts, setname='B')
    loader_B = torch.utils.data.DataLoader(dataset_B, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    # directory
    result_dir = os.path.join(opts.result_dir, opts.name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # test
    print('\n--- testing B ---')
    metric_setup = Metrics()
    for index, (image, label, names) in enumerate(loader_B):
        image, label, label_name = image.cuda(), label.cuda(), names[1]
        with torch.no_grad():
            predict = model.segmentor.forward_b(image)
        predict_mask = model.predict_mask(predict)
        _, h, w = label.shape
        predict_mask = translation(predict_mask, h, w)
        one_hot_label = model.one_hot_coding(label)
        ont_hot_predict = model.one_hot_coding(predict_mask)
        diceB = metric_setup.dice_coef_multilabel(one_hot_label, ont_hot_predict, opts.n_classes)
        metric_setup.update()
        print('{}/{}, {}, dice:{}'.format(index, len(loader_A), names[1], diceB))
    metric_B = metric_setup.mean()
    print('diceB:{}'.format(metric_B['dice']))

    print('\n--- testing A ---')
    metric_setup = Metrics()
    for index, (image, label, names) in enumerate(loader_A):
        image, label, label_name = image.cuda(), label.cuda(), names[1]
        with torch.no_grad():
            predict = model.segmentor.forward_a(image)
        predict_mask = model.predict_mask(predict)
        _, h, w = label.shape
        predict_mask = translation(predict_mask, h, w)
        one_hot_label = model.one_hot_coding(label)
        one_hot_predict = model.one_hot_coding(predict_mask)
        diceA = metric_setup.dice_coef_multilabel(one_hot_label, one_hot_predict, opts.n_classes)
        metric_setup.update()
        print('{}/{}, {}, dice:{}'.format(index, len(loader_A), names[1], diceA))
    metric_A = metric_setup.mean()
    print('diceB:{}'.format(metric_B['dice']))
    print('diceA:{}'.format(metric_A['dice']))

if __name__ == '__main__':
    main()
