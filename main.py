import os
import time
import argparse
#import logging
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from scipy.stats import gmean
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import prepare_folders
from train import Exp


#from resnet import resnet50
# from fcnet import fcnet1
# from loss import *
# from datasets import BostonHousing
# from utils import *

os.environ["KMP_WARNINGS"] = "FALSE"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# CPU only
parser.add_argument('--cpu_only', action='store_true', default=False, help='whether to use CPU only')
# imbalanced related
# LDS
parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

# FDS
parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
parser.add_argument('--fds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                    help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

# re-weighting: SQRT_INV / INV
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')

# training/optimization related
parser.add_argument('--dataset', type=str, default='rubbingmura', choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str, default='./housing.data', help='data directory')
parser.add_argument('--model', type=str, default='LR', choices=['LR', 'FCN'], help='model name')
parser.add_argument('--store_root', type=str, default='checkpoint', help='root path for storing checkpoints, logs')
parser.add_argument('--store_name', type=str, default='', help='experiment store name')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber'], help='training loss type')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
#parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--batch_size', type=int, nargs='+', default=[64, 64, 64], help='batch size')
parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
#### new! workers default=0, if workers>0, then shut down.
parser.add_argument('--workers', type=int, default=0, help='number of workers used in data loading')
# checkpoints
parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')

parser.set_defaults(augment=True)
args, unknown = parser.parse_known_args()

args.cpu_only = True # Use CPU to train/test models
# Option 1: To train the basic model, use the default setting, don't need to do anything
# Option 1: Evaluate the basic model
# args.reweight = 'none'
# args.lds = False
args.evaluate = False
# args.resume = './trained_models/ckpt.base.pth.tar'

args.store_name = ''
args.start_epoch, args.best_loss = 0, 1e5

if len(args.store_name):
    args.store_name = f'_{args.store_name}'
if not args.lds and args.reweight != 'none':
    args.store_name += f'_{args.reweight}'
if args.lds:
    args.store_name += f'_lds_{args.lds_kernel[:3]}_{args.lds_ks}'
    if args.lds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.lds_sigma}'
if args.fds:
    args.store_name += f'_fds_{args.fds_kernel[:3]}_{args.fds_ks}'
    if args.fds_kernel in ['gaussian', 'laplace']:
        args.store_name += f'_{args.fds_sigma}'
    args.store_name += f'_{args.start_update}_{args.start_smooth}_{args.fds_mmt}'
args.store_name = f"{args.dataset}_{args.model}{args.store_name}_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"

prepare_folders(args)

print(f"Args: {args}")
print(f"Store name: {args.store_name}")

exp = Exp(args)
exp.train()