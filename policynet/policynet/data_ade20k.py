import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import hflip, resize, crop
from torchvision.transforms import InterpolationMode


class TokenSharingPolicyDatasetADE20K(Dataset):
  def __init__(self,
               cfg,
               split='training',
               mode='training',
               apply_hflip=True,
               resize_dim=512):
    assert split in ['training', 'validation'], 'Only train and test splits are implemented.'

    gt_dir = os.path.join(os.environ['DATASET'], cfg.gt_dir_ade20k)
    assert os.path.exists(gt_dir), 'gt_dir path does not exist: {}'.format(gt_dir)

    self.split = split
    self.hflip = apply_hflip
    self.resize = resize_dim

    if mode == 'training':
      self.training = True
    else:
      self.training = False

    self.gt_dir_img = os.path.join(gt_dir, 'images', split)
    self.gt_dir_ann = os.path.join(gt_dir, 'annotations', split)

    self.gt_img_ids = [fn.replace('.jpg', '') for fn in os.listdir(self.gt_dir_img)]

    self.mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    self.std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)

  # Returns the length of the dataset
  def __len__(self):
    return len(self.gt_img_ids)

  # Returns a dataset sample given an idx [0, len(dataset))
  def __getitem__(self, idx):
    img_id = self.gt_img_ids[idx]

    image_path = os.path.join(self.gt_dir_img, img_id + '.jpg')
    image = read_image(image_path)

    label_path = os.path.join(self.gt_dir_ann, img_id + '.png')
    label = read_image(label_path)
    label = torch.as_tensor(label, dtype=torch.long).squeeze(0)

    # Normalize the image
    image = torch.as_tensor(image, dtype=torch.float)

    image = image - self.mean
    image = image / self.std

    image = torch.as_tensor(image, dtype=torch.float)

    # Resize
    image = resize(image.unsqueeze(0), (self.resize, self.resize), interpolation=InterpolationMode.BILINEAR).squeeze(0)
    label = resize(label.unsqueeze(0), (self.resize, self.resize), interpolation=InterpolationMode.NEAREST).squeeze(0)

    if self.training and self.hflip:
      flip = random.uniform(0, 1)
      if flip > 0.5:
        image = hflip(image)
        label = hflip(label)

    label = policy_gt_gen(label,
                          patch_size=16,
                          group_if_all_ignore=True,
                          ignore_if_one_class_plus_ignore=False,
                          ignore_if_one_class_plus_ignore_border=True)

    return image, label, image_path


def policy_gt_gen(gt,
                  patch_size,
                  group_if_all_ignore=True,
                  ignore_if_one_class_plus_ignore=False,
                  ignore_if_one_class_plus_ignore_border=False,
                  ):
  H, W = gt.size()
  patch_size = patch_size * 2

  gt = gt.unsqueeze(0)

  gt[gt == 0] = 255

  max = F.max_pool2d(gt.float(), patch_size, stride=patch_size).to(torch.int32)
  min = F.max_pool2d(-gt.float(), patch_size, stride=patch_size).to(torch.int32)
  one_class = (max == -min)

  gt_tmp = gt.clone().detach().float()
  gt_tmp[gt == 255] = -1000
  max_new = F.max_pool2d(gt_tmp, patch_size, stride=patch_size).to(torch.int32)
  one_class_and_maybe_ignore = (max_new == -min)

  one_class_plus_ignore = torch.logical_and(torch.logical_not(one_class), one_class_and_maybe_ignore)

  patch_groups_per_img = one_class.to(torch.uint8)
  one_class_plus_ignore = one_class_plus_ignore.to(torch.uint8)

  ignore_present = max == 255

  if not group_if_all_ignore:
    ignore = torch.logical_and(ignore_present, one_class)
    patch_groups_per_img[ignore] = 255

  if ignore_if_one_class_plus_ignore:
    patch_groups_per_img[one_class_plus_ignore] = 255

  if ignore_if_one_class_plus_ignore_border:
    border = torch.zeros_like(one_class_plus_ignore)

    border[0, 0, :] = True
    border[0, -1, :] = True
    border[0, :, 0] = True
    border[0, :, -1] = True

    one_class_plus_ignore_border = torch.logical_and(one_class_plus_ignore, border)
    patch_groups_per_img[one_class_plus_ignore_border] = 255

  return patch_groups_per_img.to(torch.long).squeeze(0)