import numpy as np
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms as T
from random import random, choice
from torch import logical_and as LAND
from torch import logical_or as LOR
# from convert import slice_model
from stacked import Stacked
from pytorch_lightning.callbacks import ModelCheckpoint
import tensorboard
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
    
    return scan, structure, structure_3d

  def __len__(self):
    self.border = int(self.len * 0.95)
    if self.mode == 'train':
      return self.border
    elif self.mode == 'test':
      return int(self.len * 0.05)

  def __getitem__(self, idx):
    if self.mode == 'train':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx])['arr_0'])
      image = torch.permute(image, (2, 0, 1)).float()
      structure = torch.from_numpy(np.array([a.T for a in np.load(self.dir + 'structures/' + self.strs[idx])['arr_0']])).float()
      structure_3d = torch.from_numpy(np.array([a.T for a in np.load(self.dir + '3d_structures/' + self.struc_3d[idx])['arr_0']])).float()
      image, label, structure_3d = self.transform(scan=image, structure=structure, structure_3d=structure_3d)
      return image, label, structure_3d

    elif self.mode == 'test':
      image = torch.from_numpy(np.load(self.dir + 'scans/' + self.scans[idx+self.border])['arr_0'])
      image = torch.permute(image, (2, 0, 1)).float()
      structure = torch.from_numpy(np.array([a.T for a in np.load(self.dir + 'structures/' + self.strs[idx+self.border])['arr_0']])).float()
      structure_3d = torch.from_numpy(np.array([a.T for a in np.load(self.dir + '3d_structures/' + self.struc_3d[idx+self.border])['arr_0']])).float()
      return image, structure, structure_3d

BATCH_SIZE = 8
device = torch.device('cuda')

def slice_model(data_2d):
    x = torch.clone(data_2d)
    x[1] = x[1] + x[0]
    
    fill_3d = torch.zeros(165, 128, 128).to(device)

    for i in range(165):
        fill_3d[i] += (x[0]-i) > 0
        fill_3d[i] += (x[1]-i) > 0
    
    return fill_3d

def calc_val_data(pred_3d, orig_3d, n_cls=3):
        
        bat_size = pred_3d.shape[0]

        intersection = torch.tensor([[torch.sum(LAND((pred_3d[b]==i),(orig_3d[b]==i))) for i in range(n_cls)] for b in range(bat_size)])
        union =        torch.tensor([[torch.sum(LOR( (pred_3d[b]==i),(orig_3d[b]==i))) for i in range(n_cls)] for b in range(bat_size)])
        target =       torch.tensor([[torch.sum(orig_3d[b]==i) for i in range(n_cls)] for b in range(bat_size)])
        
        # Output shapes: batch_size x num_classes
        return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):

    mean_iou = torch.mean((intersection+eps)/(union+eps))
    mean_class_rec = torch.mean((intersection+eps)/(target+eps))
    mean_acc = torch.nansum(intersection)/torch.nansum(target)

    return mean_iou, mean_class_rec, mean_acc

def calc_miou(pred_3d, orig_3d, n_cls=3):
    intersection, union, target = calc_val_data(pred_3d, orig_3d, n_cls=3)
    mean_iou, mean_class_rec, mean_acc = calc_val_loss(intersection, union, target, eps = 1e-7)
    return mean_iou, mean_class_rec, mean_acc


class SomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Stacked(n_stacks=3).to(device)

    def forward(self, x):
        x.to(device)
        return self.model(x)

    

    def training_step(self, batch, batch_idx):
        x, y, y3d = batch
        x = x.to(device)
        y = y.to(device)

        pred = self(x)
        loss = sum([F.mse_loss(inter, y) for inter in pred])/3
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, y3d = batch
        x = x.to(device)
        y = y.to(device)
        # loss = F.mse_loss(self(x), y)
        pred = self(x)
        loss = sum([F.mse_loss(inter, y) for inter in pred])/3
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        #miou
        sliced_preds = torch.stack([slice_model(y_pred_i) for y_pred_i in pred[-1]])
        sliced_trues = torch.stack([slice_model(y_true_i) for y_true_i in y])

        # print(sliced_preds.shape)

        mean_iou, mean_class_rec, mean_acc = calc_miou(sliced_preds, sliced_trues, n_cls=3)
        
        self.log("mean_iou", mean_iou, prog_bar=True)
        self.log("mean_class_rec", mean_class_rec, prog_bar=True)
        self.log("mean_acc", mean_acc, prog_bar=True)
        # return loss
        return {'MSE': loss, 'pred': pred[-1], 'mask': y}

    def validation_epoch_end(self, outputs):
        mse = torch.tensor([x['MSE'] for x in outputs])

        # mean_iou, mean_class_rec, mean_acc = loss.calc_val_loss(intersection, union, target, self.eps)
        # mean_iou, mean_class_rec, mean_acc = calc_val_loss(intersection, union, target, self.eps)
        mean_mse = torch.mean(mse)

        log_dict = {'mean_mse': mean_mse}

        for k, v in log_dict.items():
            self.log(k, v, prog_bar=True)

         # Visualize results
        # img = torch.cat([x['img'] for x in outputs]).cpu()
        pred_1 = torch.cat([x['pred'][0][0] for x in outputs]).unsqueeze(dim=0).cpu()
        pred_2 = torch.cat([x['pred'][0][1] for x in outputs]).unsqueeze(dim=0).cpu()

        mask_1 = torch.cat([x['mask'][0][0] for x in outputs]).unsqueeze(dim=0).cpu()
        mask_2 = torch.cat([x['mask'][0][1] for x in outputs]).unsqueeze(dim=0).cpu()


        # pred_vis = self.visualize_mask(torch.argmax(pred, dim=1))
        # mask_vis = self.visualize_mask(mask)
        results = torch.cat(torch.cat([pred_1, pred_2, mask_1, mask_2], dim=2).split(1, dim=0), dim=1)
        # results_thumbnail = F.interpolate(results, scale_factor=0.25, mode='bilinear')[0]

        self.logger.experiment.add_image('results', results, self.current_epoch)
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
    
    
    def setup(self, stage=None):
        
        self.train_set, self.test_set, self.val_set = random_split(struct_dataset, [len(struct_dataset)-200, 100, 100])
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            pass
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            pass
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCH_SIZE, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=BATCH_SIZE, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=BATCH_SIZE, num_workers=8)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

if __name__ == "__main__":
    struct_dataset = CustomData('dataset2/', mode='train')
    train_set, val_set = torch.utils.data.random_split(struct_dataset, [len(struct_dataset)-200, 200])
    some_model = SomeModel().to(device)
    logger = pl.loggers.TensorBoardLogger(save_dir=f'logs', name="archive_stacked_4_3_mean_100_metrics")



    # saves model every 2d epoch
    checkpoint_callback = ModelCheckpoint(every_n_epochs=10,
                                      save_top_k = -1,
                                      filename="archive_stacked_4_3_mean_100_metrics-{epoch:02d}",
                                      dirpath ='checkpoints/'
                                      )
    # Initialize a trainer
    trainer = Trainer(
        devices=[0], 
        accelerator="gpu",
        max_epochs=100,
        progress_bar_refresh_rate=1,
        logger=logger,
        callbacks = checkpoint_callback
    )

    # Train the model ⚡
    trainer.fit(some_model)