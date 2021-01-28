import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random

class dataset_segmentation(data.Dataset):
    def __init__(self, opts, phase, setname):
        self.opts = opts
        self.phase = phase
        self.dataroot = opts.dataroot
        self.setname = setname
        self.mask_code = {0: 0, 60: 1, 120: 2, 180: 3}
        images = os.listdir(os.path.join(self.dataroot, self.phase + setname))
        self.img = [os.path.join(self.dataroot, self.phase + setname, x) for x in images]
        self.size = len(self.img)

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('%s: %d images' % (setname, self.size))

    def __getitem__(self, index):
        image_path_name = self.img[index]
        image_name = os.path.split(self.img[index])[-1]
        if self.setname == 'A':
            label_name = 'mr_label' + image_name.split('image')[-1]
        elif self.setname == 'B':
            label_name = 'ct_label' + image_name.split('image')[-1]
        label_path_name = os.path.join(self.dataroot, self.phase + self.setname + '_label', label_name)

        image, label = self.load(image_path_name, label_path_name)

        return image, label, (image_name, label_name)

    def load(self, img_path_name, lab_path_name):

        img = Image.open(img_path_name).convert('RGB')
        lab = Image.open(lab_path_name)
        input = self.transforms(img)
        lab_arr = np.array(lab)
        remap_target = lab_arr.copy()
        for k, v in self.mask_code.items():
            remap_target[lab_arr == k] = v
        target = torch.from_numpy(remap_target).long()

        return input, target

    def __len__(self):
        return self.size

class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.dataroot = opts.dataroot
        self.mask_code = {0: 0, 60: 1, 120: 2}

        # A
        images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
        # self.A_dict = self.slice_dict(images_A)
        self.A_label_path = os.path.join(self.dataroot, opts.phase + 'A_label')

        # B
        images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
        self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]
        # self.B_dict = self.slice_dict(images_B)
        self.B_label_path = os.path.join(self.dataroot, opts.phase + 'B_label')

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = min(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        # if not opts.no_flip:
        #   transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return

    def __getitem__(self, index):

        data_A_name = os.path.split(self.A[index])[-1]
        PBS_B = os.path.join(self.B[random.randint(0, self.B_size - 1)])
        A_label_name = 'mr_label' + data_A_name.split('image')[-1]
        B_label_name = 'ct_label' + os.path.split(PBS_B)[-1].split('image')[-1]

        if self.opts.phase == 'train':
            data_A, data_A_lab = self.load(self.A[index], os.path.join(self.A_label_path, A_label_name))
            data_B, data_B_lab = self.load(PBS_B, os.path.join(self.B_label_path, B_label_name))
            return data_A, data_A_lab, data_B, data_B_lab
        elif self.opts.phase == 'test':
            A_image_name = data_A_name
            B_image_name = os.path.split(PBS_B)[-1]
            data_A, data_A_lab, A_raw_size = self.load(self.A[index], os.path.join(self.A_label_path, A_label_name))
            data_B, data_B_lab, B_raw_size = self.load(PBS_B, os.path.join(self.B_label_path, B_label_name))
            return data_A, data_B, data_A_lab, data_B_lab, (A_image_name, A_label_name), (B_image_name, B_label_name)

    def slice_select(self, A_case_index, B_case_index, slice_index):
        k_mri, k_ct = len(self.A_dict[A_case_index]) - 1, len(self.B_dict[B_case_index]) - 1
        m = random.randint(-5, 5)
        out_slice_index = int(int(slice_index) * (k_ct / k_mri))
        if out_slice_index >= 5 and out_slice_index < (k_ct - 5):
            out_slice_index = out_slice_index + m
        else:
            out_slice_index = out_slice_index
        if out_slice_index == 0:
            out_slice_index = 1
        return out_slice_index

    def slice_dict(self, MRI_list):
        case_candidate = []
        for i in range(len(MRI_list)):
            case_candidate.append(MRI_list[i].split('_')[2])
        case_number = list(set(case_candidate))

        slice_dict = {}
        for j in range(len(MRI_list)):
            for k in range(len(case_number)):
                if MRI_list[j].split('_')[2] == case_number[k]:
                    slice_dict.setdefault(case_number[k], []).append(MRI_list[j])

        return slice_dict

    def load(self, img_name, lab_name):

        img = Image.open(img_name).convert('RGB')
        lab = Image.open(lab_name)
        if not self.opts.no_flip:
            if random.random() < 0.5:
                img, lab = img.transpose(Image.FLIP_LEFT_RIGHT), lab.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img, lab = img.transpose(Image.FLIP_TOP_BOTTOM), lab.transpose(Image.FLIP_TOP_BOTTOM)
        input = self.transforms(img)

        if self.opts.phase == 'train':
            lab = lab.resize((self.opts.resize_size, self.opts.resize_size), Image.NEAREST)

        lab_arr = np.array(lab)
        remap_target = lab_arr.copy()
        for k, v in self.mask_code.items():
            remap_target[lab_arr == k] = v
        target = torch.from_numpy(remap_target).long()

        return input, target

    def __len__(self):
        return self.dataset_size


# call dataset
if __name__ == "__main__":
    from options import TrainOptions

    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    dataset = dataset_unpair(opts)

    print('done')
