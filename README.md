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
Download our test data (link-> https://duke.box.com/s/04b6788r4jxve309byuepq2nb1j990ko) to the folder named Data.
Download our pre-trained models (link-> https://duke.box.com/s/m5bpfgn29if6413okmz7nchpsswywd0o) to the folder named Models.

Apply our Pre-trained Model with GPU

Download our test data to the data folder.

python Main.py 


Citation

If you use this code for your research, please cite our paper:




