#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   validation.py
@Time    :   2020/12/01 14:02:51
@Author  :   Runze Wang
@Contact :   runze.wang@sjtu.edu.cn
'''

# here put the import lib
import os
import wandb
import torch
from model import DRIT
from options import TestOptions
from dataset import dataset_pair
from evaluation import Metrics
from saver import *

wandb.init(project="drit")

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = dataset_pair(opts)
    if opts.a2b:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    resume_files = os.listdir(opts.resume_path)
    resume_files.sort(key=lambda ele:ele.split('.')[0])
    for file in resume_files:
        if 'pth' in file:
            epoch = int(file.split('.')[0])
            print('Resume file from {}'.format(file))
            model.resume(os.path.join(opts.resume_path, file), train=False)
            model.eval()

            # validation
            print('\n--- validating ---')
            metric = Metrics()
            # import ipdb;ipdb.set_trace()
            for idx, (img_a, img_b) in enumerate(loader):
                MR = img_a.cuda(opts.gpu)
                CT = img_b.cuda(opts.gpu)
                with torch.no_grad():
                    CT_out = model.test_forward_transfer(MR, CT, a2b=True)
                target_arr = tensor2img(CT)
                target_img = Image.fromarray(target_arr).convert('L')
                target = np.array(target_img)
                output_arr = tensor2img(CT_out)
                output_img = Image.fromarray(output_arr).convert('L')
                output_resize = output_img.resize(target_img.size, Image.BICUBIC)
                output = np.array(output_resize)
                mae, mse, psnr, ssim = metric.compute(target, output)
                print('{}/{}, mae:{}, psnr:{}, ssim:{}'.format(idx, len(loader),mae, psnr, ssim))
                metric.update(mae, mse, psnr, ssim)
            result = metric.mean()
            wandb.log({'epoch': epoch, 'mae': result.get('mae'), 'psnr': result.get('psnr'), 'ssim': result.get('ssim')})

            print('epoch:{}, mae:{}, psnr:{}, ssim:{}'.
                    format(epoch, result.get('mae'), result.get('psnr'), result.get('ssim')))
            print('Done')

if __name__ == '__main__':
    main()








