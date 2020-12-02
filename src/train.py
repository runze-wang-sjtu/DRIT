import torch
import wandb
import numpy as np
from PIL import Image
from options import TrainOptions
from dataset import dataset_unpair, dataset_pair
from model import DRIT
from saver import Saver, tensor2img
from evaluation import Metrics

wandb.init(project="drit")

def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # daita loader
    print('\n--- load training dataset ---')
    train_dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                                num_workers=opts.nThreads)
    print('\n--- load validation dataset ---')
    val_dataset = dataset_pair(opts, phase_name='test')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)

    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        # train
        print('\n--- train ---')
        model.train()
        for it, (images_a, images_b) in enumerate(train_loader):
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue
            # input data
            images_a = images_a.cuda(opts.gpu).detach()
            images_b = images_b.cuda(opts.gpu).detach()

            # update model
            if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                model.update_D_content(images_a, images_b)
                continue
            else:
                model.update_D(images_a, images_b)
                model.update_EG()

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()
        # save result image
        saver.write_img(ep, model)
        # Save network weights
        saver.write_model(ep, total_it, model)

        # eval
        print('\n--- eval ---')
        metric = Metrics()
        model.eval()
        for idx, (img_a, img_b) in enumerate(val_loader):
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
            print('{}/{}, mae:{}, psnr:{}, ssim:{}'.format(idx, len(val_loader),mae, psnr, ssim))
            metric.update(mae, mse, psnr, ssim)
        result = metric.mean()
        wandb.log({'epoch': ep, 'mae': result.get('mae'), 'psnr': result.get('psnr'), 'ssim': result.get('ssim')})

        print('epoch:{}, mae:{}, psnr:{}, ssim:{}'.
                format(ep, result.get('mae'), result.get('psnr'), result.get('ssim')))

if __name__ == '__main__':
    main()
