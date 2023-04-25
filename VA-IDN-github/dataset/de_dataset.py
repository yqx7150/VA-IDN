from __future__ import print_function, division
import os, random, time

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import rawpy
from glob import glob
from PIL import Image as PILImage
import numbers
from scipy.misc import imread
import scipy.io as io
import cv2


class deDataset(Dataset):

    def __init__(self, opt, root1, root2):
    
        self.task = opt.task
        self.root2 = root2

        target_forward = np.array([root2 +"/"+ x  for x in os.listdir(root2)])
        input_data = np.array([root1 +"/"+ x  for x in os.listdir(root1)])

        self.data = {'input_data':input_data, 'target_forward':target_forward}

    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
        
    def random_flip(self, input_raw, target_rgb):
        idx = np.random.randint(2)
        input_raw = np.flip(input_raw,axis=idx).copy()
        target_rgb = np.flip(target_rgb,axis=idx).copy()
        
        return input_raw, target_rgb

    def random_rotate(self, input_raw, target_rgb):
        idx = np.random.randint(4)
        input_raw = np.rot90(input_raw,k=idx)
        target_rgb = np.rot90(target_rgb,k=idx)

        return input_raw, target_rgb

    def random_crop(self, patch_size, input_raw, target_rgb):
        
        H = input_raw.shape[0]
        W = input_raw.shape[1]
        rnd_h = random.randint(0, max(0, H - patch_size))
        rnd_w = random.randint(0, max(0, W - patch_size))
        
        if len(input_raw.shape) == 2:
            patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size]
        else:    
            
            patch_input_raw = input_raw[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
            
        if len(target_rgb.shape) == 2:

            patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size]
        else:
            patch_target_rgb = target_rgb[rnd_h:rnd_h + patch_size, rnd_w:rnd_w + patch_size, :]
        
        return patch_input_raw, patch_target_rgb
        
    def aug(self, patch_size, input_raw, target_rgb):
        input_raw, target_rgb = self.random_crop(patch_size, input_raw,target_rgb)
        #input_raw, target_rgb = cv2.resize(input_raw, (patch_size, patch_size) ) , cv2.resize(target_rgb, (patch_size, patch_size)  )
        input_raw, target_rgb = self.random_rotate(input_raw,target_rgb)
        input_raw, target_rgb = self.random_flip(input_raw,target_rgb)
        
        return input_raw, target_rgb
       
        
    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def __len__(self):
        return len(self.data['target_forward'])

    def __getitem__(self, idx):
        
        self.data['input_data'].sort()
        self.data['target_forward'].sort()
        
        input_path = self.data['input_data'][idx]
        #target_forward_path = self.data['target_forward'][idx]  
        target_forward_path = os.path.join(self.root2, input_path.split('/')[-1])    
            
        if not os.path.exists(input_path) and not os.path.exists(target_forward_path):
            assert False
        
        assert (input_path.split('/')[-1]) == (target_forward_path.split('/')[-1])
        
        input_img = cv2.imread(input_path)  
        target_forward_img_aug = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)[:, :, 0]  # ( , )        
                
        if self.task == 'train':
        
            self.patch_size = 256
            input_img_aug, target_forward_img_aug = self.aug(self.patch_size, input_img, target_forward_img_aug) 
            
        else:
            input_img_aug, target_forward_img_aug = input_img, target_forward_img_aug  

        target_forward_img_aug = self.norm_img(target_forward_img_aug, max_value=255)
        input_ch = self.norm_img(input_img_aug, max_value=255) 
        

        h,w = target_forward_img_aug.shape

        target_forward_img_aug = np.expand_dims(target_forward_img_aug, 2) # (, , 1)


        target_forward_img = np.zeros((h,w,3))
        target_forward_img[:,:,0] = target_forward_img_aug[:,:,0]
        target_forward_img[:,:,1] = target_forward_img_aug[:,:,0]
        target_forward_img[:,:,2] = target_forward_img_aug[:,:,0]  # 64 64 3
        
        
        input_target_img = input_ch.copy()


        input_ch = self.np2tensor(input_ch).float()
        target_forward_img = self.np2tensor(target_forward_img).float()
        input_target_img = self.np2tensor(input_target_img).float()

        sample = {'input_ch':input_ch, 'target_forward_img':target_forward_img, 'input_target_img':input_target_img,
                    'file_name':target_forward_path.split("/")[-1].split(".")[0]}
        return sample


