# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright Skoltech Deep Learning Course.

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import tensorboard
import pytorch_lightning as pl
import time
import torch

from UNet import UNet
from metrics_miou import *
from Dataloader import *
# from .model import UNet, DeepLab
# from .dataset import FloodNet
# from . import loss



class SegModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = None,
        batch_size: int = 16,
        data_path: str = 'dataset128_32',
        image_size: int = 128,
    ):
        super().__init__()
        self.net = UNet()
        
        self.train_dataset = TomographySet(data_path+"/", 'train') 
        # data_size = len(self.train_dataset)
        # tenth = 0.1 * data_size
        # self.train_dataset, _ = random_split(self.train_dataset, [tenth, data_size - tenth], generator=torch.Generator().manual_seed(0))
        
        self.test_dataset = TomographySet(data_path+"/", 'test')
        # _, self.train_dataset, _ = random_split(self.test_dataset, [tenth, tenth, data_size - tenth * 2], generator=torch.Generator().manual_seed(0))

        self.batch_size = batch_size
        self.lr = lr
        self.eps = 1e-7

        # Visualization
        self.color_map = torch.FloatTensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
             [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)
        mask[0][0] *= 15
        pred[0][1] *= 150

        train_loss = F.mse_loss(pred, mask)

        self.log('train_loss', train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img)
        mask[0][0] *= 15
        pred[0][1] *= 150
        # intersection, union, target = loss.calc_val_data(pred, mask, self.num_classes)
        # intersection, union, target = calc_val_data(pred, mask)
        mse = F.mse_loss(pred, mask)

        return {'MSE': mse, 'pred': pred, 'mask': mask}

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
    # def visualize_mask(self, mask):

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=20)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=8, batch_size=1, shuffle=False)


def define_model(lr: float, 
                 checkpoint_name: str = '', 
                 batch_size: int = 16):
    assignment_dir = 'stacked_res'
    experiment_name = f'Exp1'
    
    model = SegModel(
        lr,
        batch_size, 
        image_size=128)

    if checkpoint_name:
        model.load_state_dict(torch.load(f'{assignment_dir}/logs/{experiment_name}/{checkpoint_name}')['state_dict'])
    
    return model, experiment_name

def train(model, experiment_name, use_gpu):
    assignment_dir = 'semantic_segmentation'

    logger = pl.loggers.TensorBoardLogger(save_dir=f'logs', name=experiment_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='mean_mse',
        dirpath=f'{assignment_dir}/logs/{experiment_name}',
        filename='{epoch:02d}-{mean_iou:.3f}',
        mode='max')
    
    trainer = pl.Trainer(
        max_epochs=100, 
        devices=[2], 
        accelerator="gpu",
        benchmark=True, 
        check_val_every_n_epoch=5, 
        logger=logger, 
        callbacks=[checkpoint_callback])

    time_start = time.time()
    
    trainer.fit(model)
    
    torch.cuda.synchronize()
    time_end = time.time()
    
    training_time = (time_end - time_start) / 60
    
    return training_time

if __name__ == "__main__":
    model, experiment_name = define_model(lr=1e-3) # experiment to find the best LR
    training_time = train(model, experiment_name, use_gpu=True)

    print(f'Training time: {training_time:.3f} minutes')