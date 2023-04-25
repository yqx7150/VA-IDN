import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage

from model.model import InvISPNet
from dataset.de_dataset import deDataset
from config.config import get_arguments

from tqdm import tqdm
import cv2
import imageio
from skimage.measure import compare_psnr, compare_ssim, compare_mse, shannon_entropy,compare_nrmse
from matplotlib import pyplot as plt
import math
import scipy.io as io


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
parser.add_argument("--dataset", type=str, default="Ncd", help="dataset name. ")

args = parser.parse_args()
print("Parsed arguments: {}".format(args))

ckpt_allname = args.ckpt.split("/")[-1]

def save_img(img, img_path):
    img = np.clip(img*255,0,255)
    cv2.imwrite(img_path, img)

def save_img_color(img, img_path):
    img = np.clip(img*255,0,255)
    
    img_1 = img[:, :, :: -1]
    #cv2.imwrite(img_path, img_1)
    cv2.imwrite(img_path, img)

def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    device = torch.device("cuda:0")
    
    net.to(device)
    net.eval()
    # load the pretrained weight if there exists one
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    else:
        assert 0
    
    print("[INFO] Start data load and preprocessing") 

    Dataset = deDataset(opt=args,root1='./dataset/color',root2='./dataset/gray')
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)


    PSNR=[]
    PSNR_REV=[]

    SSIM=[]
    SSIM_REV=[]

    MSE=[]
    NMSE=[]
    
    NRMSE=[]
    
    TIME=[]

    print("[INFO] Start test...") 
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time() 

        input, target_forward, input_target = sample_batched['input_ch'].to(device), sample_batched['target_forward_img'].to(device), \
                            sample_batched['input_target_img'].to(device)

        file_name = sample_batched['file_name'][0]        


        with torch.no_grad():
            reconstruct_for = net(input)
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)

            reconstruct_rev = net(reconstruct_for, rev=True)

        pred_rev = reconstruct_rev.detach().permute(0,2,3,1).squeeze()  
        pred_rev = torch.clamp(pred_rev, 0, 1).cpu().numpy() 
        
        pred_img = reconstruct_for.detach().permute(0,2,3,1).squeeze().cpu().numpy()   
        target_forward_img = target_forward.permute(0,2,3,1).squeeze().cpu().numpy() 
        
        pred_for = ( pred_img[:,:,0] + pred_img[:,:,1] + pred_img[:,:,2] ) / 3.0
        target_forward_patch = ( target_forward_img[:,:,0] + target_forward_img[:,:,1] + target_forward_img[:,:,2] ) / 3.0
               
        target_rev_patch = input_target.permute(0,2,3,1).squeeze().cpu().numpy()  
                   
        target_rev = target_rev_patch


        psnr = compare_psnr( 255 * abs(target_forward_patch),255 * abs(pred_for), data_range=255)
        psnr_rev = compare_psnr( 255 * abs(target_rev),255 * abs(pred_rev), data_range=255)
        ssim = compare_ssim(abs(target_forward_patch), abs(pred_for), data_range=1,multichannel=True)
        ssim_rev = compare_ssim(abs(target_rev), abs(pred_rev), data_range=1,multichannel=True)

        mse = compare_mse(target_forward_patch,pred_for)

        nmse =  np.sum((pred_for - target_forward_patch) ** 2.) / np.sum(target_forward_patch**2)

        PSNR.append(psnr)
        PSNR_REV.append(psnr_rev)

        SSIM.append(ssim)
        SSIM_REV.append(ssim_rev)

        MSE.append(mse)

        NMSE.append(nmse)

        save_path= 'exps/{}/test/{}'.format(args.dataset, ckpt_allname)
        
        os.makedirs(save_path+'/pred', exist_ok=True)
        os.makedirs(save_path+'/pred_mat', exist_ok=True)
        
        os.makedirs(save_path+'/target', exist_ok=True)     
        os.makedirs(save_path+'/target_mat', exist_ok=True)    
          
        os.makedirs(save_path+'/pred_rev', exist_ok=True)    
        os.makedirs(save_path+'/pred_rev_mat', exist_ok=True)

        os.makedirs(save_path+'/target_rev', exist_ok=True)
        os.makedirs(save_path+'/target_rev_mat', exist_ok=True)

        save_img(pred_for, save_path+'/pred'+'/pred_'+file_name+'.png')
        io.savemat(save_path+'/pred_mat'+'/pred_'+file_name+'.mat',{'data':pred_for})
        
        save_img(target_forward_patch, save_path+'/target'+'/target_'+file_name+'.png')
        io.savemat(save_path+'/target_mat'+'/target_'+file_name+'.mat',{'data':target_forward_patch})
        
        save_img_color(pred_rev, save_path+'/pred_rev'+'/pred_rev_'+file_name+'.png')  
        io.savemat(save_path+'/pred_rev_mat'+'/pred_rev_'+file_name+'.mat',{'data':pred_rev})

        save_img_color(target_rev, save_path+'/target_rev'+'/target_rev_'+file_name+'.png')  
        io.savemat(save_path+'/target_rev_mat'+'/target_rev_'+file_name+'.mat',{'data':target_rev})
        
        times =  time.time()-step_time
        
        TIME.append(times)
        
        print("[INFO] Epoch time: ", time.time()-step_time, "task: ", args.task)

        del reconstruct_for
        del reconstruct_rev
        
    ave_time = sum(TIME) / len(TIME)
    all_time = sum(TIME)

    ave_psnr = sum(PSNR) / len(PSNR)
    PSNR_std = np.std(PSNR)
    
    ave_psnr_rev = sum(PSNR_REV) / len(PSNR_REV)
    PSNR_REV_std = np.std(PSNR_REV)
    
    
    ave_ssim = sum(SSIM) / len(SSIM)
    SSIM_std = np.std(SSIM)
    
    ave_ssim_rev = sum(SSIM_REV) / len(SSIM_REV)
    SSIM_REV_std = np.std(SSIM_REV)
    
    ave_mse = sum(MSE) / len(MSE)
    
    ave_nmse = sum(NMSE) / len(NMSE)
    NMSE_std = np.std(NMSE)
    

    print('ave_psnr',ave_psnr)
    print('ave_psnr_rev',ave_psnr_rev)

    print('ave_ssim',ave_ssim)
    print('ave_ssim_rev',ave_ssim_rev)

    print('ave_mse',ave_mse)
    print('ave_nmse',ave_nmse)
    
    with open('results_test.txt', 'a+') as f:
        f.write('\n'*3)
        f.write(ckpt_allname+'\n')
        
        f.write('ave_time:'+str(ave_time)+' '*3+'all_time:'+str(all_time)+'\n')
                
        f.write('ave_psnr:'+str(ave_psnr)+' '*3+'PSNR_std:'+str(PSNR_std)+'\n')

        f.write('ave_psnr_rev:'+str(ave_psnr_rev)+' '*3+'PSNR_REV_std:'+str(PSNR_REV_std)+'\n')
  
    
        f.write('ave_ssim:'+str(ave_ssim)+' '*3+'SSIM_std:'+str(SSIM_std)+'\n')

        f.write('ave_ssim_rev:'+str(ave_ssim_rev)+' '*3+'SSIM_REV_std:'+str(SSIM_REV_std)+'\n')

        f.write('ave_mse:'+str(ave_mse)+'\n')
        

        f.write('ave_nmse:'+str(ave_nmse)+' '*3+'NMSE_std:'+str(NMSE_std)+'\n')
        


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

