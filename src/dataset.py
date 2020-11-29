import os
import torch
import numpy as np
import SimpleITK as sitk
from numpy.lib.arraypad import pad
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, transforms
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
    print('%s: %d images'%(setname, self.size))
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

class dataset_25D(data.Dataset):
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
    if not self.opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('CT: %d, MR: %d images'%(self.CT_size, self.MR_size))

  def __getitem__(self, index):
    MR_3d = sitk.ReadImage(os.path.join(self.opts.dataroot, self.opts.phase+'MR', self.MR[index]))
    CT_3d = sitk.ReadImage(os.path.join(self.opts.dataroot, self.opts.phase+'CT', self.CT[index]))
    MR_array = sitk.GetArrayFromImage(MR_3d)
    CT_array = sitk.GetArrayFromImage(CT_3d)
    MR_x, CT_x = self.slice_select(MR_array.shape[0], CT_array.shape[0])
    MR_y, CT_y = self.slice_select(MR_array.shape[1], CT_array.shape[1])
    MR_z, CT_z = self.slice_select(MR_array.shape[2], CT_array.shape[2])
    pad_target = max(max(MR_array.shape),max(CT_array.shape))

    MR_slice_x = self.pad(MR_array[MR_x,:,:], pad_target)
    MR_slice_y = self.pad(MR_array[:,MR_y,:], pad_target)
    MR_slice_z = self.pad(MR_array[:,:, MR_z], pad_target)
    CT_slice_x = self.pad(CT_array[CT_x,:,:], pad_target)
    CT_slice_y = self.pad(CT_array[:,CT_y,:], pad_target)
    CT_slice_z = self.pad(CT_array[:,:, CT_z], pad_target)

    MR_array_25d = np.array([MR_slice_x, MR_slice_y, MR_slice_z]).transpose(1,2,0)
    CT_array_25d = np.array([CT_slice_x, CT_slice_y, CT_slice_z]).transpose(1,2,0)
    MR = self.transforms(MR_array_25d)
    CT = self.transforms(CT_array_25d)

    return MR, CT
  
  def caculate_pad_number(self, input, target):
    if (target - input)%2==0:
      pad_number_1 = pad_number_2 = int((target-input)/2)
    else:
      pad_number_1 = int((target-input+1)/2-1)
      pad_number_2 = int((target-input+1)/2)
    return pad_number_1, pad_number_2
  
  def pad(self, array, target_size):
    x1, x2 = self.caculate_pad_number(array.shape[0], target_size)
    y1, y2 = self.caculate_pad_number(array.shape[1], target_size)
    output = np.pad(array, ((x1, x2), (y1, y2)), 'constant', constant_values = ((0,0),(0,0)))
    return output

  def slice_select(self, MR_dim, CT_dim):
    x = random.random()
    m = random.randint(-5, 5)
    MR_x = int(x*MR_dim)
    if MR_x > 5 and MR_x < (MR_dim-5):
      CT_x = int(x*CT_dim) + m
    else:
      CT_x = int(x*CT_dim)
    return MR_x, CT_x

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

# call dataset
if __name__ == "__main__":

    from options import TrainOptions
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    dataset = dataset_25D(opts)
    mr, ct = dataset.__getitem__(1)

    print('done')

