# This file is modified from
# https://github.com/guochengqian/openpoints/blob/2bc0bf9cb2aee0fcd61f6cdc3abca1207e5e809e/dataset/s3dis/s3dis.py
import os
import gdown
import pickle
import numpy as np
from tqdm import tqdm
from typing import Literal
from pathlib import Path
from torch.utils.data import Dataset
from . import transforms as T
from collections import namedtuple

url = "https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y"
DATA_DIR = Path(__file__).parent / 'data'  # code/dataset/data


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


def download_s3dis():
    DATA_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / 's3disfull').exists():
        if not (DATA_DIR / 's3disfull.tar').exists():
            print('Downloading S3DISFull dataset...')
            gdown.download(url, DATA_DIR / 's3disfull.tar', quiet=False)
        print('Extracting...')
        os.system(f'tar -xvf {DATA_DIR / "s3disfull.tar"} -C {DATA_DIR}')


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * \
                 np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


VoxelGrid = namedtuple('VoxelGrid', ['p2v', 'v2p', 'v2p_start', 'v_pcount'])


def voxelize(coord, voxel_size=0.05, hash_type='fnv'):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, p_sort2voxel_id, voxel_pcount = np.unique(key_sort, return_counts=True, return_inverse=True)
    p2voxel_id = np.empty(len(coord), dtype=np.int64)
    p2voxel_id[idx_sort] = p_sort2voxel_id

    v2point_id = idx_sort  # voxel to point id
    v2point_id_start = np.cumsum(np.insert(voxel_pcount, 0, 0)[0:-1])  # voxel to point id start
    return VoxelGrid(p2voxel_id, v2point_id, v2point_id_start, voxel_pcount)


def voxel_select(voxel: VoxelGrid, start_idx=None):
    if exists(start_idx):
        offsets = start_idx % voxel.v_pcount
    else:
        offsets = np.random.randint(0, voxel.v_pcount.max(), voxel.v_pcount.size) % voxel.v_pcount
    pid_select = voxel.v2p_start + offsets
    return pid_select


def crop_pc(coord, feat, label, voxel_max, init_idx=None, variable=True, shuffle=True):
    crop_idx = None
    N = len(label)  # the number of points
    if N >= voxel_max:
        init_idx = default(init_idx, np.random.randint(N))
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
    elif not variable:
        # fill more points for non-variable case (batched data)
        cur_num_points = N
        query_inds = np.arange(cur_num_points)
        padding_choice = np.random.choice(cur_num_points, voxel_max - cur_num_points)
        crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
    crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
    if shuffle:
        np.random.shuffle(crop_idx)
    coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[
        crop_idx] if label is not None else None

    coord -= coord.min(0)

    return (coord.astype(np.float32),
            feat.astype(np.float32) if feat is not None else None,
            label.astype('i8') if label is not None else None)


def combine_features(xyz, rgb, height, feature):
    if feature == 'xyz':
        return xyz
    elif feature == 'xyzrgb':
        return np.concatenate([xyz, rgb], axis=-1)
    elif feature == 'rgbh':
        return np.concatenate([rgb, height], axis=-1)
    else:
        raise NotImplementedError


S3DIS_data = namedtuple('S3DIS_data', ['xyz', 'feat', 'label', 'idx_parts'])


class S3DIS(Dataset):
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    num_classes = 13
    num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                              650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    class2color = {'ceiling': [0, 255, 0],
                   'floor': [0, 0, 255],
                   'wall': [0, 255, 255],
                   'beam': [255, 255, 0],
                   'column': [255, 0, 255],
                   'window': [100, 100, 255],
                   'door': [200, 200, 100],
                   'table': [170, 120, 200],
                   'chair': [255, 0, 0],
                   'sofa': [200, 100, 100],
                   'bookcase': [10, 200, 100],
                   'board': [200, 200, 200],
                   'clutter': [50, 50, 50]}
    cmap = [*class2color.values()]
    gravity_dim = 2
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """

    def __init__(self,
                 data_root: str = DATA_DIR / 's3disfull',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 feature: Literal['xyz', 'xyzrgb', 'rgbh'] = 'rgbh',
                 split: str = 'train',
                 transform: T.Transform = T.TransformCompose([T.ColorNormalize(), T.XYZAlign()]),
                 loop: int = 1,
                 presample: bool = False,
                 variable: bool = False,
                 shuffle: bool = True,
                 ):
        download_s3dis()

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.feature = feature
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        data_root = Path(data_root)
        raw_root = data_root / 'raw'
        self.raw_root = raw_root
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if split == 'train':
            self.fns = [item for item in data_list if 'Area_{}'.format(test_area) not in item]
        else:
            self.fns = [item for item in data_list if 'Area_{}'.format(test_area) in item]

        processed_root = data_root / 'processed'
        filename = processed_root / f's3dis_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl'
        # presample is only for validation
        if presample and not os.path.exists(filename):
            self.data = []
            for item in tqdm(self.fns, desc=f'Loading S3DISFull {split} split on Test Area {test_area}'):
                # room data
                data_path = raw_root / (item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    xyz, rgb, label = np.split(cdata, [3, 6], axis=-1)
                    voxel = voxelize(xyz, voxel_size)
                    uniq_idx = voxel_select(voxel, 0)
                    xyz, rgb, label = xyz[uniq_idx], rgb[uniq_idx], label[uniq_idx]
                    cdata = np.hstack([xyz, rgb, label])
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            print('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))

            # dump
            processed_root.mkdir(exist_ok=True)
            pickle.dump(self.data, open(filename, 'wb'))
            print(f"{filename} saved successfully")
        elif presample:
            # load
            self.data = pickle.load(open(filename, 'rb'))
            print(f"{filename} load successfully")

        assert len(self.fns) > 0
        print(f"\nTotally {len(self.fns)} samples in {split} set")

    def __getitem__(self, idx):
        if self.split == 'test':
            # for test
            return self.get_test_data(idx)

        xyz, rgb, label = self.get_data(idx)
        label = label.squeeze(-1).astype('i8')
        data = {'xyz': xyz, 'rgb': rgb, 'label': label}
        # pre-process.
        if exists(self.transform):
            data = self.transform(T.Inputs(**data))

        # to float32
        xyz = data['xyz'].astype(np.float32).copy()
        height = xyz[:, [self.gravity_dim]]
        rgb = data['rgb'].astype(np.float32).copy()
        label = data['label'].astype(np.int64).copy()

        feat = combine_features(xyz, rgb, height, self.feature)

        return S3DIS_data(xyz=xyz, feat=feat, label=label, idx_parts=[])

    def __len__(self):
        return len(self.fns) * self.loop

    def get_data(self, idx):
        idx = idx % len(self.fns)
        if self.presample:
            data = self.data[idx]
            xyz, rgb, label = np.split(data, [3, 6], axis=-1)
        else:
            data_path = os.path.join(self.raw_root, self.fns[idx] + '.npy')
            data = np.load(data_path).astype(np.float32)
            data[:, :3] -= np.min(data[:, :3], 0)
            xyz, rgb, label = np.split(data, [3, 6], axis=-1)

            # voxelize
            if exists(self.voxel_size):
                voxel = voxelize(xyz, self.voxel_size)
                uniq_idx = voxel_select(voxel, None)
                xyz, rgb, label = map(lambda x: x[uniq_idx], [xyz, rgb, label])
            xyz, rgb, label = crop_pc(xyz, rgb, label, self.voxel_max, init_idx=None, variable=self.variable,
                                      shuffle=self.shuffle)
        return xyz, rgb, label

    def get_test_data(self, idx):
        # for test
        idx = idx % len(self.fns)
        data = np.load(self.raw_root / (self.fns[idx] + '.npy')).astype(np.float32)
        data[:, :3] -= np.min(data[:, :3], 0)
        xyz, rgb, label = data[:, :3], data[:, 3:6], data[:, 6]

        idx_parts = []
        if exists(self.voxel_size):
            # idx_sort: original point indicies sorted by voxel NO.
            # voxel_idx: Voxel NO. for the sorted points
            voxel = voxelize(xyz, self.voxel_size)
            for i in range(voxel.v_pcount.max()):
                pid_select = voxel_select(voxel, i)
                idx_parts.append(pid_select)
        else:
            idx_parts.append(np.arange(label.shape[0]))

        if exists(self.transform):
            _data = self.transform(T.Inputs(xyz=xyz, rgb=rgb, label=label))
            xyz, rgb, label = _data['xyz'], _data['rgb'], _data['label']

        height = xyz[:, [self.gravity_dim]]
        feat = combine_features(xyz, rgb, height, self.feature)
        return S3DIS_data(xyz=xyz.astype('f4'), feat=feat.astype('f4'), label=label.astype('i8'), idx_parts=idx_parts)


if __name__ == '__main__':
    S3DIS()[0]
