import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms as T
from random import random

class TomographySet(Dataset):

  def __init__(self, dir, mode):
    self.mode = mode
    self.dir = dir
    self.scans = [x for x in os.listdir(self.dir + 'scans') if x[-3:]=='npy']
    self.strs = os.listdir(self.dir + 'structures')
    self.len = len(self.scans)

  @staticmethod
  def transform(image, label):
  
    # Horizontal flip
    if random()>0.5:
      image = T.functional.hflip(image)
      label = T.functional.hflip(label)
    
    # Vertical flip
    if random()>0.5:
      image = T.functional.vflip(image)
      label = T.functional.vflip(label)
  
    # Rotation
    rand_deg = T.RandomRotation.get_params(degrees=[-30, 30])
    image = T.functional.rotate(image, angle = rand_deg)
    label = T.functional.rotate(label, angle = rand_deg)
    
    return image, label

  def __len__(self):
    self.border = int(self.len * 0.8)
    if self.mode == 'train':
      return self.border
    elif self.mode == 'test':
      return int(self.len * 0.2)

  def __getitem__(self, idx):
    if self.mode == 'train':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx]))
      image = torch.permute(image, (2, 0, 1))
      label = torch.from_numpy(np.load(self.dir + 'structures/' + self.strs[idx]))
      image, label = transform(image=image, label=label)
      assert image.shape == torch.Size([31, 128, 128]) and label.shape == torch.Size([2, 128, 128])
      return image.float(), label.float()

    elif self.mode == 'test':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx+self.border]))
      image = torch.permute(image, (2, 0, 1))
      label = torch.from_numpy(np.load(self.dir + 'structures/' + self.strs[idx+self.border]))
      assert image.shape == torch.Size([31, 128, 128]) and label.shape == torch.Size([2, 128, 128])
      return image.float(), label.float()
    

# Initialization (here PATH_TO_DATA is path to directory where scans and structures are located, e.g. '/content/drive/MyDrive/dataset0/', 'MODE' = 'test' or 'train'):
loader = DataLoader(TomographySet('PATH_TO_DATA', mode='MODE'), num_workers=2, batch_size=16, shuffle=True)
