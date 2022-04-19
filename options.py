import argparse
import os
import time
import shutil
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--epoch', type=int, default=301, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='./pre_train/resnet50-19c8e357.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='E://pytorch/data/RGBDcollection_fast/RGB/', help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='E://pytorch/data/RGBDcollection_fast/depth/', help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='E://pytorch/data/RGBDcollection_fast/GT/', help='the training gt images root')
parser.add_argument('--edge_root', type=str, default='E://pytorch/data/RGBDcollection_fast/edge/', help='the training edge images root')
parser.add_argument('--test_rgb_root', type=str, default='E://pytorch/data/test_in_train/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='E://pytorch/data/test_in_train/depth/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default='E://pytorch/data/test_in_train/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='./results/train', help='the path to save models and logs')
opt = parser.parse_args()

