# JA-GAN: Joint Attention GAN for Multimodal Image Translation

The Pytorch implements of our ACA-GAN: Autoencoder-based Collaborative Attention GAN for Multi-modal Image Synthesis.

<img src="https://pic-liujiaxu.oss-cn-beijing.aliyuncs.com/20210121200111.png"/>

## Setup
1. Conda create -n ja-gan python=3.7
2. Install following requirements via pip or conda.
```
python              3.7.12
pytorch             1.12.1
torchvision         0.13.0
tqdm                4.64.1
numpy               1.21.6
SimpleITK           2.1.1.1
opencv-python       4.6.0.66
easydict            1.9
tensorboard         2.11.0
Pillow              9.2.0
```
## Prepare datasets
1. Download BRATS2020 (https://www.med.upenn.edu/cbica/brats2020/data.html)
2. Rearrange file to the following structure
```
MICCAI_BraTS2020_TrainingData
├── flair
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_002_flair.nii.gz
│   ├── BraTS20_Training_003_flair.nii.gz
│   ├── ...
├── t2
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_002_t2.nii.gz
│   ├── BraTS20_Training_003_t2.nii.gz
│   ├── ...
├── t1
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_002_t1.nii.gz
│   ├── BraTS20_Training_003_t1.nii.gz
│   ├── ...
├── t1ce
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_002_t1ce.nii.gz
│   ├── BraTS20_Training_003_t1ce.nii.gz
│   ├── ...
```
## Train
```
python train.py options/brats/joint_attention.yaml
```

## Test
```
python test.py options/brats/joint_attention.yaml
```

## Acknowledge
This project is based on Junyan Zhu's outstanding implementation of [CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)