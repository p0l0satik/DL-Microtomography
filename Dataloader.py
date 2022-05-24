import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms as T
from random import random, choice

# For .npz data    
class CustomData(Dataset):

  def __init__(self, dir, mode):
    self.mode = mode
    self.dir = dir
    self.scans = sorted([x for x in os.listdir(self.dir + 'scans') if x[-3:]=='npz'])
    self.strs = sorted(os.listdir(self.dir + 'structures'))
    self.struc_3d = sorted([x for x in os.listdir(self.dir + '3d_structures') if x[-3:]=='npz'])
    self.len = len(self.scans)


  @staticmethod
  def transform(scan, structure, structure_3d):
  
    # Horizontal flip
    if random()>0.5:
      scan = T.functional.hflip(scan)
      structure = T.functional.hflip(structure)
      structure_3d = T.functional.hflip(structure_3d)

    
    #Vertical flip
    if random()>0.5:
      scan = T.functional.vflip(scan)
      structure = T.functional.vflip(structure)
      structure_3d = T.functional.vflip(structure_3d)

  
    #Rotation
    if random()>0.5:
      rand_deg = choice([90, -90])
      scan = T.functional.rotate(scan, angle = rand_deg)
      structure = T.functional.rotate(structure, angle = rand_deg)
      structure_3d = T.functional.rotate(structure_3d, angle = rand_deg)
    
    return image, label

  def __len__(self):
    self.border = int(self.len * 0.8)
    if self.mode == 'train':
      return self.border
    elif self.mode == 'test':
      return int(self.len * 0.2)

  def __getitem__(self, idx):
    if self.mode == 'train':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx])['arr_0'])
      image = torch.permute(image, (2, 0, 1)).float()
      label = torch.from_numpy(np.load(self.dir + 'structures/' + self.strs[idx])['arr_0']).float()
      structure_3d = torch.from_numpy(np.load(self.dir + '3d_structures/' + self.struc_3d[idx])['arr_0'].T).float()
      image, label = self.transform(image=image, label=label, structure_3d=structure_3d)
      return image, label, structure_3d

    elif self.mode == 'test':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx+self.border])['arr_0'])
      image = torch.permute(image, (2, 0, 1)).float()
      label = torch.from_numpy(np.load(self.dir + 'structures/' + self.strs[idx+self.border])['arr_0'].T).float()
      structure_3d = torch.from_numpy(np.load(self.dir + '3d_structures/' + self.struc_3d[idx+self.border])['arr_0'].T).float()
      return image, label, structure_3d

# Initialization (here PATH_TO_DATA is path to directory where scans and structures are located, e.g. '/content/drive/MyDrive/dataset0/', 'MODE' = 'test' or 'train'):
loader = DataLoader(TomographySet('PATH_TO_DATA', mode='MODE'), num_workers=2, batch_size=16, shuffle=True)
