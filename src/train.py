import torch
import wandb
import numpy as np
from options import TrainOptions
from dataset import dataset_unpair, dataset_segmentation
from model import DRIT
from saver import Saver
from metric import Metrics
from PIL import Image


def translation(x, h, w):
    x_local = x.squeeze().cpu().numpy()
    x_Image = Image.fromarray(x_local.astype('uint8'))
    x_resize = np.array(x_Image.resize((w, h), Image.NEAREST))
    x_cuda = torch.from_numpy(x_resize).unsqueeze(0).cuda()
    return x_cuda

def main():

    wandb.init(project="heart-segmentation")    # parse options

    parser = TrainOptions()
    opts = parser.parse()

    # daita loader
    print('\n--- load training dataset ---')
    train_dataset = dataset_unpair(opts)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)

    print('\n --- load test dataset ---')
    test_dataset_A = dataset_segmentation(opts, phase='test', setname='A')
    test_loader_A = torch.utils.data.DataLoader(test_dataset_A, batch_size=1, num_workers=opts.nThreads)
    test_dataset_B = dataset_segmentation(opts, phase='test', setname='B')
    test_loader_B = torch.utils.data.DataLoader(test_dataset_B, batch_size=1, num_workers=opts.nThreads)


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
        for it, (images_a, labels_a, images_b, labels_b) in enumerate(train_loader):
            if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
                continue

            # input data
            images_a = images_a.cuda(opts.gpu).detach()
            images_b = images_b.cuda(opts.gpu).detach()
            labels_a = labels_a.cuda(opts.gpu).detach()
            labels_b = labels_b.cuda(opts.gpu).detach()

            # update model
            if ep < opts.n_ep_before_seg:
                if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                    model.update_D_content(images_a, images_b)
                    continue
                else:
                    model.update_D(images_a, images_b)
                    model.update_EG()
            else:
                if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                    model.update_D_content(images_a, images_b)
                    continue
                else:
                    model.update_D(images_a, images_b)
                    model.update_EG()
                    seg_loss = model.update_Seg(labels_a, labels_b)
                    wandb.log(seg_loss)
                    # wandb.log(train_dice)

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

        # # save result image
        saver.write_img(ep, model)

        # # Save network weights
        saver.write_model(ep, total_it, model)


        if ep >= opts.n_ep_before_seg:

            model.eval()
            print('\n--- testing B ---')
            metric_setup = Metrics()
            for index, (image, label, names) in enumerate(test_loader_B):
                image, label, label_name = image.cuda(opts.gpu), label.cuda(opts.gpu), names[1]
                with torch.no_grad():
                    predict = model.segmentor.forward_b(image)
                predict_mask = model.predict_mask(predict)
                _, h, w = label.shape
                predict_mask = translation(predict_mask, h, w)
                one_hot_label = model.one_hot_coding(label, cuda=False)
                ont_hot_predict = model.one_hot_coding(predict_mask, cuda=False)
                _ = metric_setup.dice_coef_multilabel(one_hot_label, ont_hot_predict, opts.n_classes)
                metric_setup.update()
            metric_B = metric_setup.mean()
            metric_B['diceB_0'] = metric_B.pop('dice_0')
            metric_B['diceB_1'] = metric_B.pop('dice_1')
            metric_B['diceB_2'] = metric_B.pop('dice_2')
            metric_B['diceB_3'] = metric_B.pop('dice_3')
            metric_B['diceB'] = metric_B.pop('dice')

            print('\n--- testing A ---')
            metric_setup = Metrics()
            for index, (image, label, names) in enumerate(test_loader_A):
                image, label, label_name = image.cuda(opts.gpu), label.cuda(opts.gpu), names[1]
                with torch.no_grad():
                    predict = model.segmentor.forward_a(image)
                predict_mask = model.predict_mask(predict)
                _, h, w = label.shape
                predict_mask = translation(predict_mask, h, w)
                one_hot_label = model.one_hot_coding(label, cuda=False)
                one_hot_predict = model.one_hot_coding(predict_mask, cuda=False)
                _ = metric_setup.dice_coef_multilabel(one_hot_label, one_hot_predict, opts.n_classes)
                metric_setup.update()
            metric_A = metric_setup.mean()
            metric_A['diceA_0'] = metric_A.pop('dice_0')
            metric_A['diceA_1'] = metric_A.pop('dice_1')
            metric_A['diceA_2'] = metric_A.pop('dice_2')
            metric_A['diceA_3'] = metric_A.pop('dice_3')
            metric_A['diceA'] = metric_A.pop('dice')

            wandb.log(metric_B)
            wandb.log(metric_A)
            wandb.log({'epoch': ep})


if __name__ == '__main__':
    main()
