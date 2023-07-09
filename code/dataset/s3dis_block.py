# This dataset uses the data used in PointNet, which samples the S3DIS rooms into blocks with area 1m by 1m.
# Each block is randomly sampled into 4096 points.
import os
import subprocess
import numpy as np
import h5py
import torch
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset

from . import transforms as T
from .s3dis import S3DIS_data
from collections import namedtuple

url = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
DATA_DIR = Path(__file__).resolve().parent / 'data'


def exists(val):
    return val is not None


def download_s3dis():
    DATA_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / 'indoor3d_sem_seg_hdf5_data').exists():
        zipfile = Path(url).name
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('indoor3d_sem_seg_hdf5_data', DATA_DIR))
        os.system('rm %s' % zipfile)
    if not (DATA_DIR / 'Stanford3dDataset_v1.2_Aligned_Version').exists():
        zippath = DATA_DIR / 'Stanford3dDataset_v1.2_Aligned_Version.zip'
        assert zippath.exists(), 'Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under %s' % DATA_DIR
        os.system('unzip %s' % zippath)
        os.system('mv %s %s' % ('Stanford3dDataset_v1.2_Aligned_Version', DATA_DIR))
        os.system('rm %s' % zippath)


def prepare_test_data_semseg():
    if not (DATA_DIR / 'stanford_indoor3d').exists():
        subprocess.run('python prepare_data/collect_indoor3d_data.py', shell=True, cwd=DATA_DIR.parent)
    if not (DATA_DIR / 'indoor3d_sem_seg_hdf5_data_test').exists():
        subprocess.run('python prepare_data/gen_indoor3d_h5.py', shell=True, cwd=DATA_DIR.parent)


def load_data_semseg(partition, test_area):
    download_s3dis()
    prepare_test_data_semseg()
    if partition == 'train':
        data_dir = DATA_DIR / 'indoor3d_sem_seg_hdf5_data'
    else:
        # for validation
        data_dir = DATA_DIR / 'indoor3d_sem_seg_hdf5_data_test'

    with open(data_dir / "all_files.txt") as f:
        all_files = [line.rstrip() for line in f]
    with open(data_dir / "room_filelist.txt") as f:
        room_filelist = [line.rstrip() for line in f]

    data_batchlist, label_batchlist = [], []
    for f in all_files:
        with h5py.File(DATA_DIR / f, 'r') as file:
            data = file["data"][:]
            label = file["label"][:]
            data_batchlist.append(data)
            label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, axis=0)
    label_batches = np.concatenate(label_batchlist, axis=0)

    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    if partition == 'train':
        all_data = data_batches[train_idxs]
        all_labels = label_batches[train_idxs]
    else:
        all_data = data_batches[test_idxs]
        all_labels = label_batches[test_idxs]
    return all_data, all_labels


def combine_features(xyz, rgb, height, feature):
    if feature == 'xyz':
        return xyz
    elif feature == 'xyzrgb':
        return np.concatenate([xyz, rgb], axis=-1)
    elif feature == 'rgbh':
        return np.concatenate([rgb, height], axis=-1)
    else:
        raise NotImplementedError


class S3DISBlock(Dataset):
    num_points = 4096  # each block has 4096 points
    gravity_dim = 2

    def __init__(self,
                 feature: Literal['xyz', 'xyzrgb', 'rgbh'] = 'rgbh',
                 partition: Literal['train', 'val', 'test'] = 'train',
                 test_area='5',
                 transform: T.Transform = None):
        self.feature = feature
        self.partition = partition
        self.transform = transform

        # load data
        self.data, self.labels = load_data_semseg(partition, test_area)

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]
        if self.partition == 'train':
            indices = list(range(data.shape[0]))
            np.random.shuffle(indices)
            data = data[indices]
            label = label[indices]

        xyz = data[:, :3]
        rgb = data[:, 3:6]

        if exists(self.transform):
            data = self.transform(T.Inputs(xyz=xyz, rgb=rgb, label=label))
            xyz = data['xyz'].astype('f4')
            rgb = data['rgb'].astype('f4')
            label = data['label'].astype('i8')

        height = xyz[:, [self.gravity_dim]]
        feat = combine_features(xyz, rgb, height, self.feature)

        return S3DIS_data(xyz=xyz, feat=feat, label=label, idx_parts=None)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    S3DISBlock()[0]
