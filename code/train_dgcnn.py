import os
from pprint import pprint
import fire

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from x_dgcnn import DGCNN_Seg
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

from dataset.s3dis import S3DIS, load_data
from dataset import transforms as T


class LitModel(pl.LightningModule):

    def __init__(
            self,
            # ---- data ----
            loop,
            test_area,
            voxel_max,
            # ---- train ----
            epochs,
            batch_size,
            lr,
            optimizer,
            weight_decay,
            warm_up,
            loss,
            label_smoothing,
            # ---- model -----
            k,
            dropout,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = DGCNN_Seg(k=k, in_dim=4, out_dim=13, dropout=dropout)

        # metrics
        self.iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)
        self.val_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)
        self.test_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)

    def forward(self, x, xyz):
        return self.net(x, xyz)

    def training_step(self, batch, batch_idx):
        xyz = batch.xyz.transpose(1, 2).contiguous()
        x = torch.cat([batch.rgb, batch.height], dim=-1).transpose(1, 2).contiguous()
        y = batch.label
        pred = self(x, xyz)
        loss = F.cross_entropy(pred, y, label_smoothing=self.hparams.label_smoothing)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        self.log('loss', loss, prog_bar=True)

        self.iou(pred, y)
        self.log('iou', self.iou, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xyz = batch.xyz.transpose(1, 2).contiguous()
        x = torch.cat([batch.rgb, batch.height], dim=-1).transpose(1, 2).contiguous()
        y = batch.label

        pred = self(x, xyz)
        loss = F.cross_entropy(pred, y, label_smoothing=self.hparams.label_smoothing)
        self.log('val_loss', loss, prog_bar=True)

        cm = self.val_iou.confmat
        oa = cm.diag().sum() / cm.sum()
        macc = cm.diag() / cm.sum(1)
        self.val_iou(pred, y)
        self.log('val_oa', oa, prog_bar=True)
        self.log('val_macc', macc.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        H = self.hparams
        if H.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=H.lr, weight_decay=H.weight_decay)
        elif H.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=H.lr, momentum=0.9, weight_decay=H.weight_decay)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.trainer.estimated_stepping_batches, max_lr=H.lr,
            pct_start=H.warm_up / self.trainer.max_epochs, div_factor=10, final_div_factor=100)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        H = self.hparams

        transform = T.TransformCompose([
            T.ColorAutoContrast(),
            T.ColorDrop(p=0.5),
            T.ColorNormalize(),
            T.AnisotropicScale(scale=[0.9, 1.1]),
            T.XYZAlign(),
            T.Rotate(angle=[0, 0, 60.]),
            T.Jitter(sigma=0.005, clip=0.02)
        ])
        dataset = S3DIS(voxel_max=H.voxel_max, test_area=H.test_area, split='train', transform=transform, loop=H.loop,
                        presample=False, shuffle=True)
        return DataLoader(dataset, batch_size=H.batch_size, shuffle=True, num_workers=4, drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        H = self.hparams
        transform = T.TransformCompose([
            T.ColorNormalize(),
            T.XYZAlign(),
        ])
        dataset = S3DIS(voxel_max=None, test_area=H.test_area, split='val', transform=transform,
                        presample=True, shuffle=False)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    def test_dataloader(self):
        H = self.hparams
        transform = T.TransformCompose([
            T.ColorNormalize(),
            T.XYZAlign(),
        ])
        dataset = S3DIS(voxel_max=None, test_area=H.test_area, split='test', transform=transform,
                        presample=False, shuffle=False)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


def main(
        # ---- data ----
        loop=10,
        voxel_max=20000,
        test_area=5,
        # ---- train ----
        epochs=100,
        batch_size=4,
        lr=1e-3,
        optimizer='AdamW',
        weight_decay=1e-2,
        warm_up=10,
        loss='cross_entropy',
        label_smoothing=0.2,
        gradient_clip_val=10,
        # ---- model -----
        k=20,
        dropout=0.3,
        # ---- log -----
        name='dgcnn',
        project='s3dis',
        offline=False,
):
    name = f"{name}_area{test_area}"
    pprint(locals())
    pl.seed_everything(42)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='s3dis_sem_experiments', name=name, save_dir='wandb', offline=offline)
    model = LitModel(loop=loop, voxel_max=voxel_max, epochs=epochs, batch_size=batch_size, lr=lr, optimizer=optimizer,
                     weight_decay=weight_decay, warm_up=warm_up, loss=loss, label_smoothing=label_smoothing, k=k,
                     dropout=dropout, test_area=test_area)

    callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)


if __name__ == '__main__':
    fire.Fire(main)
