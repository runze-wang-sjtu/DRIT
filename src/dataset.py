import os
import torch
import numpy as np
import SimpleITK as sitk
from numpy.lib.arraypad import pad
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomVerticalFlip, RandomHorizontalFlip, \
    ToTensor, Normalize, \
    transforms
import random


class dataset_single(data.Dataset):
    def __init__(self, opts, setname, input_dim):
        self.dataroot = opts.dataroot
        images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
        self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
        self.size = len(self.img)
        self.input_dim = input_dim

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transforms.append(CenterCrop(opts.crop_size))
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('%s: %d images' % (setname, self.size))
        return

    def __getitem__(self, index):
        data = self.load_img(self.img[index], self.input_dim)
        return data

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.size


class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.CT = os.listdir(os.path.join(self.opts.dataroot, self.opts.phase + 'CT'))
        self.MR = os.listdir(os.path.join(self.opts.dataroot, self.opts.phase + 'MR'))
        self.CT_size = len(self.CT)
        self.MR_size = len(self.MR)
        self.dataset_size = max(self.CT_size, self.MR_size)
        self.input_dim_CT = self.opts.input_dim_CT
        self.input_dim_MR = self.opts.input_dim_MR

        # setup image transformation
        transforms = []
        transforms.append(RandomCrop(opts.crop_size))
        transforms.append(RandomVerticalFlip(p=1.0))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('CT: %d, MR: %d images' % (self.CT_size, self.MR_size))

    def __getitem__(self, index):
        MR_3d = sitk.ReadImage(os.path.join(self.opts.dataroot, self.opts.phase + 'MR', self.MR[index]))
        CT_3d = sitk.ReadImage(os.path.join(self.opts.dataroot, self.opts.phase + 'CT', self.CT[index]))
        MR_array = sitk.GetArrayFromImage(MR_3d)
        CT_array = sitk.GetArrayFromImage(CT_3d)
        MR_x, CT_x = self.slice_select(MR_array.shape[2], CT_array.shape[2])

        MR_slice_x = MR_array[:, :, MR_x]
        CT_slice_x = CT_array[:, :, CT_x]

        MR_img = Image.fromarray(MR_slice_x).convert('RGB')
        CT_img = Image.fromarray(CT_slice_x).convert('RGB')
        MR = self.transforms(MR_img)
        CT = self.transforms(CT_img)
        return MR, CT

    def slice_select(self, MR_dim, CT_dim):
        if self.opts.center_part:
            MR_x = int(random.uniform(int(MR_dim * 0.08), int(MR_dim * 0.92)))
            CT_x = int(random.uniform(int(CT_dim * 0.08), int(CT_dim * 0.92)))
        else:
            coin = random.choice([0, 1])
            if coin == 0:
                MR_x = int(random.uniform(0, int(MR_dim * 0.12)))
                CT_x = int(random.uniform(0, int(CT_dim * 0.12)))
            else:
                MR_x = int(random.uniform(int(MR_dim * 0.88), MR_dim))
                CT_x = int(random.uniform(int(CT_dim * 0.88), CT_dim))
        return MR_x, CT_x

    def __len__(self):
        return self.dataset_size


# call dataset
if __name__ == "__main__":
    from options import TrainOptions

    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    dataset = dataset_unpair(opts)
    mr, ct = dataset.__getitem__(1)

    print('done')
