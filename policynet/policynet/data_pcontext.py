import os
import glob
import numpy as np
import torch
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import hflip, crop, resize
from torchvision.transforms import InterpolationMode


class TokenSharingPolicyDatasetPContext(Dataset):
  def __init__(self,
               cfg,
               split='training',
               mode='training',
               apply_hflip=True,
               crop_size=(480, 480),
               resize_dim=512):
    assert split in ['training', 'validation'], 'Only train and test splits are implemented.'

    gt_dir = os.path.join(os.environ['DATASET'], cfg.gt_dir_pascal)
    assert os.path.exists(gt_dir), 'gt_dir path does not exist: {}'.format(gt_dir)

    self.split = split
    self.hflip = apply_hflip
    self.crop_size = crop_size
    self.resize = resize_dim

    if mode == 'training':
      self.training = True
    else:
      self.training = False

    if split == 'training':
      with open(os.path.join(gt_dir, 'ImageSets/SegmentationContext/train.txt'), 'r') as txt_file:
        img_ids = txt_file.readlines()
    else:
      with open(os.path.join(gt_dir, 'ImageSets/SegmentationContext/val.txt'), 'r') as txt_file:
        img_ids = txt_file.readlines()
    self.img_ids = [img_id.replace('\n', '') for img_id in img_ids]

    self.gt_dir_img = os.path.join(gt_dir, 'JPEGImages')
    self.gt_dir_ann = os.path.join(gt_dir, 'SegmentationClassContext')

    self.gt_img_fns = glob.glob(os.path.join(self.gt_dir_img, "*/*.png"))

    self.mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    self.std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)

  # Returns the length of the dataset
  def __len__(self):
    return len(self.img_ids)

  # Returns a dataset sample given an idx [0, len(dataset))
  def __getitem__(self, idx):
    img_id = self.img_ids[idx]

    image_path = os.path.join(self.gt_dir_img, img_id + '.jpg')
    image = read_image(image_path)

    label_path = os.path.join(self.gt_dir_ann, img_id + '.png')
    label = read_image(label_path)
    label = torch.as_tensor(label, dtype=torch.long).squeeze(0)

    # Normalize the image
    image = torch.as_tensor(image, dtype=torch.float)

    image = image - self.mean
    image = image / self.std

    if torch.isinf(torch.mean(image)) or torch.isnan(torch.mean(image)):
      print("INF OR NAN: ", image_path)

    # Resize shortest side
    image = resize(image.unsqueeze(0), self.resize, interpolation=InterpolationMode.BILINEAR).squeeze(0)
    label = resize(label.unsqueeze(0), self.resize, interpolation=InterpolationMode.NEAREST).squeeze(0)

    # Crop the image
    img_h = image.shape[1]
    img_w = image.shape[2]
    max_crop_h = img_h - self.crop_size[0]
    max_crop_w = img_w - self.crop_size[1]
    crop_offset_h = random.uniform(0, 1)
    crop_offset_w = random.uniform(0, 1)
    crop_offset_h = crop_offset_h * max_crop_h
    crop_offset_w = crop_offset_w * max_crop_w

    image = crop(image,
                 top=int(crop_offset_h),
                 left=int(crop_offset_w),
                 height=self.crop_size[0],
                 width=self.crop_size[1])

    label = crop(label,
                 top=int(crop_offset_h),
                 left=int(crop_offset_w),
                 height=self.crop_size[0],
                 width=self.crop_size[1])

    if self.training and self.hflip:
      flip = random.uniform(0, 1)
      if flip > 0.5:
        image = hflip(image)
        label = hflip(label)

    label = policy_gt_gen(label,
                          patch_size=16,
                          group_if_all_ignore=True,
                          ignore_if_one_class_plus_ignore=False)

    return image, label, image_path


def policy_gt_gen(gt,
                  patch_size,
                  group_if_all_ignore=True,
                  ignore_if_one_class_plus_ignore=False,
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

  return patch_groups_per_img.to(torch.long).squeeze(0)