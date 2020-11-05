import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, transforms
import random
import ipdb

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim, change2_LR=False):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.number = len(self.img)
    self.input_dim = input_dim
    self.change2_LR = change2_LR

    # setup image transformation
    # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    # transforms.append(CenterCrop(opts.crop_size))
    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.number))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    img_name = self.img[index]
    return (data, img_name)

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    if self.change2_LR:
      (w, h) = (img.width//4, img.height//4)
      img_LR = img.resize((w, h),Image.BICUBIC)
      img_up = img_LR.resize((img.width, img.height), Image.BICUBIC)
      img = self.transforms(img_up)
    else:
      img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.number

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
      data_A = self.load_img(self.A[index], self.input_dim_A, change2LR=True)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A, change2LR=True)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim, change2LR=False):
    if change2LR:
      img = Image.open(img_name).convert('RGB')
      (w, h) = (img.width//4, img.height//4)
      img_LR = img.resize((w, h),Image.BICUBIC)
      img_up = img_LR.resize((img.width, img.height), Image.BICUBIC)
      img_input = self.transforms(img_up)
      img_HR = self.transforms(img)
      if input_dim == 1:
        img_input  = img_input[0, ...] * 0.299 + img_input[1, ...] * 0.587 + img_input[2, ...] * 0.114
        img_input = img_input.unsqueeze(0)
        img_HR  = img_HR[0, ...] * 0.299 + img_HR[1, ...] * 0.587 + img_HR[2, ...] * 0.114
        img_HR = img_HR.unsqueeze(0)
      return (img_input, img_HR)
    else:
      img = Image.open(img_name).convert('RGB')
      img = self.transforms(img)
      if input_dim == 1:
        img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
        img = img.unsqueeze(0)
      return img

  def __len__(self):
    return self.dataset_size
