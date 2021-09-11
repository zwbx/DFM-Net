import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from net import DFMNet
from data import test_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./dataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id == '3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')
elif opt.gpu_id=='all':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print('USE GPU 0,1,2,3')

#load the model
model = DFMNet()
model.load_state_dict(torch.load('./pretrain/DFMNet_epoch_300.pth'))
model.cuda()
model.eval()

#test


def save(res,gt,notation=None,sigmoid=True):
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze() if sigmoid ==True else res.data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('save img to: ', os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name))
    cv2.imwrite(os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name), res * 255)

test_datasets = ['NJU2K','NLPR','STERE', 'RGBD135', 'LFSD','SIP']
for dataset in test_datasets:
    with torch.no_grad():
        save_path = './results/benchmark/' + dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        depth_root=dataset_path +dataset +'/depth/'
        test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)

        for i in range(test_loader.size):
            image, gt,depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            torch.cuda.synchronize()
            time_s = time.time()
            out = model(image,depth)
            torch.cuda.synchronize()
            time_e = time.time()
            t = time_e - time_s
            print("time: {:.2f} ms".format(t*1000))
            save(out[0],gt)
    print('Test Done!')
