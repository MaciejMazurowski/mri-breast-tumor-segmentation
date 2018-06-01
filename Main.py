import os
from Segmenation import BreastSeg,BreastTumor
from ImageProcessing import read_image,Norm_Zscore,imgnorm,save_image,labeling_seg
import argparse
import torch
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
    
parser = argparse.ArgumentParser(description='Tumor Segmentation from DCE-MRI')
parser.add_argument('--cuda', type=int, default='1', required=False, help='Run in GPU')
parser.add_argument('--pre', type=str, default='Data/Img1_pre.nii.gz', required=False, help='Image path for pre-constrast image')
parser.add_argument('--post', type=str, default='Data/Img1_post.nii.gz', required=False, help='Image path for post-constrast image')
parser.add_argument('--outfolder',type=str,default='Results',required=False,help='Folder for saving results')
opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda 0")



savepath = opt.outfolder

if not os.path.exists(savepath):
    os.makedirs(savepath)

# read images and output spacing information   
print('Reading images')      
img_pre, img_post, img_sub, scale_subject = read_image(opt)

# perform the intensity normalization 1) remover outliers 2) Z-score normalization 
# spatial normalization will be done before segmentation in the functions of BrestSeg and BreastTumor
print('Performing image normalization')
img_pre  = Norm_Zscore(imgnorm(img_pre))
img_post = Norm_Zscore(imgnorm(img_post))
img_sub  = Norm_Zscore(imgnorm(img_sub))

# read models
# the weights of the models will be loaded in the function of BrestSeg and BreastTumor
from Models_3D import ModelBreast,ModelTumor


if opt.cuda:
    model_breast = ModelBreast(1,1).cuda()
else:
    model_breast = ModelBreast(1,1)
if opt.cuda:
    model_tumor = ModelTumor(3,1).cuda()
else:
    model_tumor = ModelTumor(3,1)
    
    
# perform image segmentation    
print('Performing image segmentation')
breast_mask = BreastSeg(img_pre,scale_subject,model_breast,opt)
prob_1st, seg_2nd = BreastTumor(img_sub,img_post,breast_mask,scale_subject,model_tumor,model_tumor,opt)

# perform labeling
labeled_image = labeling_seg(seg_2nd)

# save images
print('Saving segmentations')
save_image(breast_mask,opt,'breast_mask.nii.gz')
save_image(prob_1st,opt,'prob_1st.nii.gz')
save_image(labeled_image,opt,'seg_2nd.nii.gz')
print('Please check the segmentation results in the result folder ---> "Results"')






