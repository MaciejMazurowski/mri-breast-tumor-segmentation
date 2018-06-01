import numpy as np
import SimpleITK as sitk

def read_image(opt):
    I_pre = sitk.ReadImage(opt.pre)  
    I_post = sitk.ReadImage(opt.post)  
    img_pre = np.array(sitk.GetArrayFromImage(I_pre))
    img_post = np.array(sitk.GetArrayFromImage(I_post))
    img_sub = img_post - img_pre
    return img_pre, img_post, img_sub, np.array(I_pre.GetSpacing())



def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img


def imgnorm(N_I,index1=0.001,index2=0.001):
    N_I = N_I.astype(np.float32)
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0

    
    return N_I

def save_image(image,opt,savename):
    I = sitk.ReadImage(opt.pre)  
    Heat_image = sitk.GetImageFromArray(image, isVector=False) 
    Heat_image.SetSpacing(I.GetSpacing())
    Heat_image.SetOrigin(I.GetOrigin())
    Heat_image.SetDirection(I.GetDirection())
    sitk.WriteImage(Heat_image,opt.outfolder+'/'+savename)     
    

from skimage.morphology import closing  
from scipy import ndimage as nd   
def labeling_seg(seg):
    bw = closing(seg)
    struct = nd.generate_binary_structure(3, 3)    
    bw = nd.morphology.binary_dilation(bw,structure=struct,iterations=7)
    bw=bw.astype('uint8')
    label_image = nd.measurements.label(bw)
    label_image = label_image[0]

    label_image = seg*label_image  
    
    return label_image