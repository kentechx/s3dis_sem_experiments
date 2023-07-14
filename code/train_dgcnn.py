import os
from pprint import pprint
from collections import namedtuple
import fire

from x_dgcnn import DGCNN_Seg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from einops import repeat

from dataset.s3dis import S3DIS
from dataset import transforms as T

Metric = namedtuple('Metric', ['miou', 'oa', 'macc'])

feature_dim = {
    'xyz': 3,
    'rgbh': 4,
    'xyzrgb': 6,
}


def calc_metrics(confmat, eps=1e-5):
    tp = confmat.diag()
    union = confmat.sum(dim=0) + confmat.sum(dim=1) - tp
    iou = (tp + eps) / (union + eps)
    miou = iou.mean()

    oa = tp.sum() / confmat.sum()
    macc = (tp + eps) / (confmat.sum(dim=1) + eps)
    macc = macc.mean()

    return Metric(miou, oa, macc)


class LitModel(pl.LightningModule):

    def __init__(
            self,
            # ---- data ----
            feature,
            loop,
            test_area,
            voxel_max,
            test_voxel_max,
            # ---- train ----
            batch_size,
            lr,
            optimizer,
            weight_decay,
            warm_up,
            loss,
            label_smoothing,
            # ---- model -----
            k,
            dynamic,
            dropout,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.net = DGCNN_Seg(feature_dim[feature], 13, k, dropout=dropout, global_pooling=False, dynamic=dynamic)

        # metrics
        self.iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)
        self.val_miou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)
        self.test_miou = torchmetrics.JaccardIndex(task='multiclass', num_classes=13)
        self.test_macc = torchmetrics.Accuracy(task='multiclass', num_classes=13, average='macro')
        self.test_oa = torchmetrics.Accuracy(task='multiclass', num_classes=13)

    def forward(self, x, xyz):
        return self.net(x, xyz)

    def training_step(self, batch, batch_idx):
        xyz = batch.xyz.transpose(1, 2).contiguous()
        x = batch.feat.transpose(1, 2).contiguous()
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
        x = batch.feat.transpose(1, 2).contiguous()
        y = batch.label

        pred = self(x, xyz)
        loss = F.cross_entropy(pred, y, label_smoothing=self.hparams.label_smoothing)
        self.log('val/loss', loss, prog_bar=True)

        self.val_miou(pred, y)
        self.log('val/miou', self.val_miou, prog_bar=True, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        y = batch.label
        preds = []
        for idx_part in batch.idx_parts:
            idx_part = idx_part[0]
            xyz = batch.xyz[:, idx_part].transpose(1, 2).contiguous()
            x = batch.feat[:, idx_part].transpose(1, 2).contiguous()
            pred = self(x, xyz)
            preds.append(pred)
        idx = torch.cat(batch.idx_parts, dim=-1)  # (1, N)
        preds = torch.cat(preds, dim=-1)
        idx = repeat(idx, 'b n -> b c n', c=preds.shape[1])
        preds = torch.scatter_reduce(torch.empty((*preds.shape[:2], y.shape[1])).to(preds),
                                     2, idx, preds, reduce='mean', include_self=False)
        preds = preds.argmax(dim=1)

        self.test_miou(preds, y)
        self.test_oa(preds, y)
        self.test_macc(preds, y)
        self.log('test/miou', self.test_miou, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('test/oa', self.test_oa, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('test/macc', self.test_macc, prog_bar=True, sync_dist=True, on_epoch=True)

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
            # T.ColorAllDrop(p=0.2),
            T.ColorNormalize(),
            T.AnisotropicScale(scale=[0.9, 1.1]),
            T.XYZAlign(),
            # T.Rotate(angle=[0, 0, 60.]),
            T.Jitter(sigma=0.005, clip=0.02)
        ])
        dataset = S3DIS(voxel_max=H.voxel_max, test_area=H.test_area, feature=H.feature, split='train',
                        transform=transform, loop=H.loop, presample=False, shuffle=True)
        return DataLoader(dataset, batch_size=H.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        H = self.hparams
        transform = T.TransformCompose([
            T.ColorNormalize(),
            T.XYZAlign(),
        ])
        dataset = S3DIS(voxel_max=H.test_voxel_max, test_area=H.test_area, feature=H.feature, split='val', loop=1,
                        transform=transform, presample=True, variable=True, shuffle=False)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    def test_dataloader(self):
        H = self.hparams
        transform = T.TransformCompose([
            T.ColorNormalize(),
            T.XYZAlign(),
        ])
        dataset = S3DIS(voxel_max=H.test_voxel_max, test_area=H.test_area, feature=H.feature, split='test', loop=1,
                        transform=transform, presample=False, variable=True, shuffle=False)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)


def main(
        # ---- data ----
        feature='rgbh',
        loop=5,
        voxel_max=40000,
        test_area=5,
        test_voxel_max=200000,
        # ---- train ----
        epochs=100,
        batch_size=1,
        lr=1e-2,
        optimizer='AdamW',
        weight_decay=1e-2,
        warm_up=10,
        loss='cross_entropy',
        label_smoothing=0.,
        gradient_clip_val=0,
        # ---- model -----
        k=40,
        dynamic=True,
        dropout=0.3,
        # ---- log -----
        name='dgcnn',
        project='s3dis',
        offline=False,
        watch=False,
        test=True,
):
    name = f"{name}_area{test_area}"
    pprint(locals())
    pl.seed_everything(42)

    os.makedirs('wandb', exist_ok=True)
    logger = WandbLogger(project='s3dis_sem_experiments', name=name, save_dir='wandb', offline=offline)
    model = LitModel(feature=feature, loop=loop, voxel_max=voxel_max, test_voxel_max=test_voxel_max,
                     batch_size=batch_size, lr=lr, optimizer=optimizer, weight_decay=weight_decay, warm_up=warm_up,
                     loss=loss, label_smoothing=label_smoothing, k=k, dynamic=dynamic, dropout=dropout,
                     test_area=test_area)

    if watch:
        logger.watch(model, log='all')

    callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(logger=logger, accelerator='cuda', max_epochs=epochs, callbacks=[callback],
                         gradient_clip_val=gradient_clip_val)
    trainer.fit(model)

    # test
    if test:
        trainer.test(model)


if __name__ == '__main__':
    fire.Fire(main)
