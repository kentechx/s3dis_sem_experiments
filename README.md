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
- [PointNext](https://github.com/kentechx/pointnext)

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

The table below presents the semantic segmentation results of the models on the S3DIS dataset. The results are evaluated
on the subsampled point clouds (voxel size = 0.04) as a common practice.

| Model        | input      | 6-fold <br/>mIoU (%) | 6-fold <br/>OA (%) | area 5<br/>mIoU (%) | area 5<br/>OA (%) | device  | report                                                |
|--------------|------------|----------------------|--------------------|---------------------|-------------------|---------|-------------------------------------------------------|
| DGCNN        | rgb+height | 53.1                 | 82.2               | 47.0                | 81.5              | 4x 3090 | [report](https://api.wandb.ai/links/kd_shen/q4z92hx2) |
| PointNext-S  | rgb+height | 67.5                 | 88.1               | 65.2                | 89.0              | 1x 3090 | [report](https://api.wandb.ai/links/kd_shen/58oqu6uk) |
| PointNext-XL | rgb+height | 71.4                 | 89.2               | 68.2                | 90.1              | 1x 3090 | [report](https://api.wandb.ai/links/kd_shen/l9xelymu) |

You can reproduce the results by running the corresponding scripts in the `code` folder with default configurations.

