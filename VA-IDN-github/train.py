
import numpy as np
import os, time, random
import argparse
import json
import cv2

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torch.nn as nn

from model.model import InvISPNet
from dataset.de_dataset import deDataset
from config.config import get_arguments

from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr

import matplotlib.pyplot as plt
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

parser = get_arguments()
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
parser.add_argument("--dataset", type=str, default="Ncd", help="dataset name. ")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path+args.dataset+"/%s"%args.task, exist_ok=True)
os.makedirs(args.out_path+args.dataset+"/%s/checkpoint"%args.task, exist_ok=True)

with open(args.out_path+args.dataset+"/%s/commandline_args.yaml"%args.task , 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def save_img(img, img_path):
    img = np.clip(img*255,0,255)

    img_1 = img[:, :, :: -1]
    cv2.imwrite(img_path, img_1)
    


def main(args):
    # ======================================define the model======================================
    writer = SummaryWriter(args.out_path+args.dataset+"/%s"%args.task)
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    net.cuda()
    # load the pretrained weight if there exists one
    if args.resume:
        net.load_state_dict(torch.load(args.out_path+"%s/checkpoint/latest.pth"%args.task))
        print("[INFO] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    print("[INFO] Start data loading and preprocessing")
    Dataset = deDataset(opt=args,root1='./VOC500/train/color',root2='./VOC500/train/gray')
    dataloader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    print("[INFO] Start to train")
    step = 0


    for epoch in range(0, 300):
        epoch_time = time.time()             
        PSNR = []
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()

            input, target_for, target_input = sample_batched['input_ch'].cuda(), sample_batched['target_forward_img'].cuda(), \
                                        sample_batched['input_target_img'].cuda()

            file_name = sample_batched['file_name'][0]

            reconstruct_for = net(input)                 
            reconstruct_for = torch.clamp(reconstruct_for, 0, 1)         

            forward_loss = F.l1_loss(reconstruct_for.cuda(), target_for.cuda())            
            writer.add_scalar('forward_loss',forward_loss.item(),global_step=step)

            reconstruct_input = net(reconstruct_for, rev=True)
            reconstruct_input = torch.clamp(reconstruct_input, 0, 1)  
                                   
            rev_loss = F.l1_loss(reconstruct_input.cuda(), target_input.cuda())
            writer.add_scalar('rev_loss',rev_loss.item(),global_step=step)
                        
            loss =  forward_loss + rev_loss             
            writer.add_scalar('loss',loss.item(),global_step=step)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("task: %s Epoch: %d Step: %d || loss: %.10f rev_loss: %.10f rev_loss: %.10f  || lr: %f time: %f"%(
                args.task, epoch, step, loss.detach().cpu().numpy(), rev_loss.detach().cpu().numpy(),
                rev_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
            ))
            
            step += 1
                  

        torch.save(net.state_dict(), args.out_path+args.dataset+"/%s/checkpoint/latest.pth"%args.task)

        if epoch % 1 == 0:
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(net.state_dict(), args.out_path+args.dataset+"/%s/checkpoint/%04d.pth"%(args.task,epoch))
            print("[INFO] Successfully saved "+args.out_path+args.dataset+"/%s/checkpoint/%04d.pth"%(args.task,epoch))

        scheduler.step()   
        
        print("[INFO] Epoch time: ", time.time()-epoch_time, "task: ", args.task)    

if __name__ == '__main__':

    torch.set_num_threads(4)
    main(args)
