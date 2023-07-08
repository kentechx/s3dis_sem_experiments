# s3dis_sem_experiments

This repository contains semantic segmentation experiments conducted on the S3DIS dataset.

## Dataset

The S3DIS dataset is annotated for indoor scene segmentation. It consists of 3D point clouds of 6 large-scale indoor
areas, with 272 rooms and 13 semantic categories. The dataset is split into 6 parts, with 5 parts for training and 1
part for testing.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Experiments

The experiments are conducted on the following models:

- [DGCNN](https://github.com/kentechx/x-dgcnn)
- [PointNet2](https://github.com/kentechx/pointnet)

Training models by running the corresponding scripts in the `code` folder. For example, to train the DGCNN model with
the default configuration, run the following command:

```bash
python code/train_dgcnn.py
```

We use the processed dataset provided by [OpenPoints](https://guochengqian.github.io/PointNeXt/examples/s3dis/),
which will be downloaded automatically when running the training scripts. The models are trained on the subsampled point
clouds (voxel size = 0.04). The model achieving the best performance on validation is selected to test on the original
point clouds (not downsampled). If you encounter download errors, you can download the dataset via the browser and move
it to the `code/dataset/data/s3disfull.tar`.

## Results

The table below presents the semantic segmentation of the models on the S3DIS dataset. The results are evaluated on the
subsampled point clouds (voxel size = 0.04) as a common practice.

| Model        | input      | 6-fold mIoU (%) | OA (%) | device |
|--------------|------------|-----------------|--------|--------|
| PointNet2SSG | rgb+height |                 |        |        |
| PointNet2MSG | rgb+height |                 |        |        |
| DGCNN        | rgb+height |                 |        |        |

You can reproduce the results by running the corresponding scripts in the `code` folder with default configurations.

