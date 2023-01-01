# ACA-GAN: Autoencoder-based Collaborative Attention GAN for Multi-modal Image Synthesis

The Pytorch implements of our ACA-GAN: Autoencoder-based Collaborative Attention GAN for Multi-modal Image Synthesis.

The overview of our ACA-GAN framework.
<img src>="../images/framework_figure1.png"
<img src="https://pic-liujiaxu.oss-cn-beijing.aliyuncs.com/20210121200111.png"/>

## Environment
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
Download the datasets from the official way and rearrange the files to the following structure.
##BraTS2020
```
BraTS2020
├── MICCAI_BraTS2020_TrainingData
│   ├── flair
│       ├── BraTS20_Training_001_flair.nii.gz
│       ├── BraTS20_Training_002_flair.nii.gz
│       ├── BraTS20_Training_003_flair.nii.gz
│       ├── ...
│   ├── t1
│       ├── BraTS20_Training_001_t1.nii.gz
│       ├── BraTS20_Training_002_t1.nii.gz
│       ├── BraTS20_Training_003_t1.nii.gz
│       ├── ...
│   ├── t1ce
│       ├── BraTS20_Training_001_t1ce.nii.gz
│       ├── BraTS20_Training_002_t1ce.nii.gz
│       ├── BraTS20_Training_003_t1ce.nii.gz
│       ├── ...
│   ├── t2
│       ├── BraTS20_Training_001_t2.nii.gz
│       ├── BraTS20_Training_002_t2.nii.gz
│       ├── BraTS20_Training_003_t2.nii.gz
│       ├── ...
├── MICCAI_BraTS2020_ValidationData
│   ├── ...
```
## Train
Edit the .yaml file of the corresponding dataset for training configuration and run the following command to train.
```
python train.py options/brats/joint_attention.yaml
```

## Test
Edit the .yaml file of the corresponding dataset for testing configuration and run the following command to test.
```
python test.py options/brats/joint_attention.yaml
```
