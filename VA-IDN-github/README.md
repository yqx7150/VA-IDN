========================train=========================

python train.py --task=train --out_path="./exps/"


========================test==========================

python decolor.py --task=test --dataset='Ncd' --out_path="./exps/" --ckpt="./exps/Ncd/train/checkpoint/latest.pth"



