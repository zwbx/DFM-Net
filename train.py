import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from datetime import datetime
from net import DFMNet
from data import get_loader,test_dataset
from utils import clip_gradient, LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

#train function
def train(train_loader, model, optimizer, epoch,save_path):

    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            depths=depths.cuda()


            cur_lr = lr_scheduler(optimizer, i, epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=(epoch-1)*total_step + i)

            out,feature_r,feature_d = model(images,depths)
            loss_f = F.binary_cross_entropy_with_logits(out[0], gts)
            loss_d = F.binary_cross_entropy_with_logits(out[1], gts)


            loss = loss_f + loss_d
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step+=1
            epoch_step+=1
            loss_all+=loss.data


            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}, loss_final: {:.4f}, loss_d: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss,loss_f.data,loss_d.data ))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format( epoch, opt.epoch, i, total_step, loss.data))
                writer.add_scalar('Loss', loss.data, global_step=step)

        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch == 300:
            torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
#test function
def test(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt,depth, name,img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res,_,_  = model(image,depth)
            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum+=np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])
        mae=mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae=mae
            torch.save(model.state_dict(), save_path + '/epoch_best.pth')
        else:
            if mae<best_mae:
                best_mae=mae
                best_epoch=epoch
                torch.save(model.state_dict(), save_path+'/epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == 'all':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print('USE GPU 0,1')
    cudnn.benchmark = True

    # build the model
    model = DFMNet()
    model.cuda()
    params = model.parameters()
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),opt.lr)

    # set the path
    image_root = opt.rgb_root
    gt_root = opt.gt_root
    depth_root = opt.depth_root
    edge_root = opt.edge_root
    test_image_root = opt.test_rgb_root
    test_gt_root = opt.test_gt_root
    test_depth_root = opt.test_depth_root
    save_path = opt.save_path
    os.mkdir(save_path)
    # load data
    print('load data...')
    train_loader = get_loader(image_root, gt_root, depth_root,edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
    total_step = len(train_loader)
    lr_scheduler = LR_Scheduler('poly', opt.lr, opt.epoch, total_step)

    logging.basicConfig(filename=save_path + '/log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + '/summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch,save_path)

