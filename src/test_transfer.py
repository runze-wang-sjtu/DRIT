import torch
from options import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  datasetA = dataset_single(opts, 'A', opts.input_dim_a, change2_LR=True)
  datasetB = dataset_single(opts, 'B', opts.input_dim_b)
  if opts.a2b:
    loader = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads, shuffle=True)
  else:
    loader = torch.utils.data.DataLoader(datasetB, batch_size=1, num_workers=opts.nThreads)
    loader_attr = torch.utils.data.DataLoader(datasetA, batch_size=1, num_workers=opts.nThreads, shuffle=True)

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
  print('\n--- testing ---')
  for idx1, img1_ in enumerate(loader):
    print('{}/{}'.format(idx1, len(loader)))
    img1 = img1_[0].cuda()
    img_name = img1_[1][0].split('/')[-1].split('.')[0]
    imgs = [img1]
    names = ['input']
    for idx2, img2_ in enumerate(loader_attr):
      if idx2 == opts.num:
        break
      img2 = img2_[0].cuda()
      with torch.no_grad():
        if opts.a2b:
          img = model.test_forward_transfer(img1, img2, a2b=True)
        else:
          img = model.test_forward_transfer(img2, img1, a2b=False)
      imgs.append(img)
      names.append('output_{}'.format(idx2))
    save_imgs(imgs, names, os.path.join(result_dir, '{}'.format(img_name)))

  return

if __name__ == '__main__':
  main()
