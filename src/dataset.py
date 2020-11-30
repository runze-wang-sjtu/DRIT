from abc import abstractstaticmethod
import os
import torch
import numpy as np
import SimpleITK as sitk
from numpy.lib.arraypad import pad
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, transforms
import random

from torchvision.transforms.transforms import RandomVerticalFlip

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

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms = []
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    # else:
    #   transforms.append(RandomCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

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
    transforms.append(RandomCrop(opts.crop_size))
    transforms.append(RandomVerticalFlip(p=1.0))
    transforms.append(RandomHorizontalFlip(p=1.0))
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
    pad_target_MR = max(MR_array.shape)
    pad_target_CT = max(CT_array.shape)

    MR_slice_x = self.pad(MR_array[MR_x,:,:], pad_target_MR)
    MR_slice_y = self.pad(MR_array[:,MR_y,:], pad_target_MR)
    MR_slice_z = self.pad(MR_array[:,:, MR_z], pad_target_MR)
    CT_slice_x = self.pad(CT_array[CT_x,:,:], pad_target_CT)
    CT_slice_y = self.pad(CT_array[:,CT_y,:], pad_target_CT)
    CT_slice_z = self.pad(CT_array[:,:, CT_z], pad_target_CT)

    MR_array_25d = np.array([MR_slice_x, MR_slice_y, MR_slice_z]).transpose(1,2,0)
    CT_array_25d = np.array([CT_slice_x, CT_slice_y, CT_slice_z]).transpose(1,2,0)
    MR_img = Image.fromarray(MR_array_25d)
    CT_img = Image.fromarray(CT_array_25d)
    MR = self.transforms(MR_img)
    CT = self.transforms(CT_img)

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
    if MR_x >= MR_dim:
      MR_x = MR_dim - 1
    if CT_x >= CT_dim:
      CT_x = CT_dim - 1
    return MR_x, CT_x

  def __len__(self):
    return self.dataset_size
class dataset_pair(data.Dataset):
  def __init__(self, opts, phase_name):
    self.opts = opts
    self.CT = os.listdir(os.path.join(self.opts.dataroot, phase_name + 'CT'))
    self.MR = os.listdir(os.path.join(self.opts.dataroot, phase_name + 'MR'))
    self.CT_size = len(self.CT)
    self.MR_size = len(self.MR)
    self.dataset_size = max(self.CT_size, self.MR_size)
    self.input_dim_CT = self.opts.input_dim_CT
    self.input_dim_MR = self.opts.input_dim_MR

    # setup image transformation
    transforms = []
    transforms.append(RandomVerticalFlip(p=1.0))
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
    pad_target_MR = max(MR_array.shape)
    pad_target_CT = max(CT_array.shape)
    assert pad_target_MR >= self.opts.crop_size
    assert pad_target_CT >= self.opts.crop_size

    MR_slice_x = self.pad(MR_array[MR_x,:,:], pad_target_MR)
    MR_slice_y = self.pad(MR_array[:,MR_y,:], pad_target_MR)
    MR_slice_z = self.pad(MR_array[:,:, MR_z], pad_target_MR)
    CT_slice_x = self.pad(CT_array[CT_x,:,:], pad_target_CT)
    CT_slice_y = self.pad(CT_array[:,CT_y,:], pad_target_CT)
    CT_slice_z = self.pad(CT_array[:,:, CT_z], pad_target_CT)
    try:
      assert MR_x < MR_array.shape[0]
      assert MR_y < MR_array.shape[1]
      assert MR_z < MR_array.shape[2]
      assert CT_x < CT_array.shape[0]
      assert CT_y < CT_array.shape[1]
      assert CT_z < CT_array.shape[2]
    except: 
      print(MR_x, MR_y, MR_z, MR_array.shape)
      print(CT_x, CT_y, CT_z, CT_array.shape)

    MR_array_25d = np.array([MR_slice_x, MR_slice_y, MR_slice_z]).transpose(1,2,0)
    CT_array_25d = np.array([CT_slice_x, CT_slice_y, CT_slice_z]).transpose(1,2,0)
    MR_img = Image.fromarray(MR_array_25d)
    CT_img = Image.fromarray(CT_array_25d)
    MR = self.transforms(MR_img)
    CT = self.transforms(CT_img)

    return MR, CT

# call dataset
if __name__ == "__main__":

    from options import TrainOptions
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    dataset = dataset_25D(opts)
    mr, ct = dataset.__getitem__(1)

    print('done')

