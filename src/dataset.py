import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize, transforms
import random
import ipdb

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.number = len(self.img)
    self.input_dim = input_dim

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
    self.A_dict = self.slice_dict(images_A)

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
    self.B_dict = self.slice_dict(images_B)

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
    data_A = self.load_img(self.A[index], self.input_dim_A)
    A_case_index = self.A[index].split('/')[-1].split('_')[1]
    A_slice_index = self.A[index].split('/')[-1].split('_')[-1].split('.')[0]
    B_case_index = self.B[index].split('/')[-1].split('_')[1]
    B_slice_index = self.slice_select(A_case_index, B_case_index, A_slice_index)
    PBS_B = self.B[index].split('_')[0]+'_'+self.B[index].split('_')[1]+'_'+str(B_slice_index)+'.jpg'
    data_B = self.load_img(PBS_B, self.input_dim_B)
    return data_A, data_B
  
  def slice_select(self, A_case_index, B_case_index, slice_index):
    k_mri, k_ct = len(self.A_dict[A_case_index])-1, len(self.B_dict[B_case_index])-1
    m = random.randint(-5, 5)
    out_slice_index = int(int(slice_index) * (k_ct/k_mri))
    if out_slice_index >= 5 and out_slice_index < (k_ct-5):
      out_slice_index = out_slice_index + m
    else:
      out_slice_index = out_slice_index
    return out_slice_index

  def slice_dict(self, MRI_list):
    case_candidate= []
    for i in range(len(MRI_list)):
      case_candidate.append(MRI_list[i].split('_')[1])
    case_number = list(set(case_candidate))

    slice_dict = {}
    for j in range(len(MRI_list)):
      for k in range(len(case_number)):
        if MRI_list[j].split('_')[1] == case_number[k]:
          slice_dict.setdefault(case_number[k], []).append(MRI_list[j])
    
    return slice_dict
          
  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size

class dataset_pair(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.number = len(self.img)
    self.input_dim = input_dim

    transforms = []
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.number))
    return

  def __getitem__(self, index):
    input_name = self.img[index]
    input = self.load_img(input_name, self.input_dim)
    target_name = input_name.split('A')[0]+'B/CT'+input_name.split('T1')[-1]
    target = self.load_img(target_name, self.input_dim)
    return (input, target, input_name, target_name)

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    return img

  def __len__(self):
    return self.number

# call dataset
if __name__ == "__main__":

    from options import TrainOptions
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    dataset = dataset_pair(opts, 'A', opts.input_dim_a)

    print('done')
