Pytorch implementation for tumor segmenation from breast DCE-MRI using pre-trained models.



The code was written by Jun Zhang, Department of Radiology at Duke University.

Applications

Tumor segmentation




Prerequisites

Linux python 2.7
pytorch version 2.0.2

NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 8.0.61
                        
Getting Started

Installation

Install pytorch and dependencies from http://pytorch.org/
Install numpy,scipy,scikit-image, and SimpleITK with pip install numpy scipy scikit-image SimpleITK


cd to current folder

Apply our Pre-trained Model with GPU

python Main.py --pre Data/Img0_pre.nii.gz --post Data/Img0_post.nii.gz


Citation

If you use this code for your research, please cite our paper:




