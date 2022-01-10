from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import ADMM_net_S9, ADMM_net_S12
from utils import generate_masks, time2file_name, A, At
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "/data/jiamianw/ICCV2021/DAVIS_data/training_data" # "./generated_data" #
test_path1 = "/data/jiamianw/BIRNAT-master/from_yangliu" # "./from_yangliu" #


mask, mask_s = generate_masks(data_path)

last_train = 200
model_save_filename = '2021_12_18_19_12_41'
batch_size = 4

stage_num = 12
n_resblocks = 14
n_feats = 24


if stage_num == 9:
    network = ADMM_net_S9(n_resblocks = n_resblocks, n_feats = n_feats).cuda()
elif stage_num == 12:
    network = ADMM_net_S12(n_resblocks=n_resblocks, n_feats=n_feats).cuda()

if last_train != 0:
    network = torch.load(
        './model/' + model_save_filename +'/S{}'.format(stage_num)+ "_model_epoch_{}.pth".format(last_train))

criterion  = nn.MSELoss()
criterion.cuda()


def test(test_path, result_path):
    test_list = os.listdir(test_path)
    psnr_stage = torch.zeros(stage_num, len(test_list))
    pred = []
    for i in range(len(test_list)):
        pic = scio.loadmat(test_path + '/' + test_list[i])

        if "orig" in pic:
            pic = pic['orig']
            sign = 1
        elif "patch_save" in pic:
            pic = pic['patch_save']
            sign = 0
        elif "p1" in pic:
            pic = pic['p1']
            sign = 0
        elif "p2" in pic:
            pic = pic['p2']
            sign = 0
        elif "p3" in pic:
            pic = pic['p3']
            sign = 0
        pic = pic / 255

        pic_gt = np.zeros([pic.shape[2] // 8, 8, 256, 256])
        for jj in range(pic.shape[2]):
            if jj % 8 == 0:
                meas_t = np.zeros([256, 256])
                n = 0
            pic_t = pic[:, :, jj]
            mask_t = mask[n, :, :]

            mask_t = mask_t.cpu()
            pic_gt[jj // 8, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t.numpy(), pic_t)

            if jj == 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % 8 == 0 and jj != 7:
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        meas = torch.from_numpy(meas)
        pic_gt = torch.from_numpy(pic_gt)
        meas = meas.cuda()
        pic_gt = pic_gt.cuda()
        meas = meas.float()
        pic_gt = pic_gt.float()
        batch_size1 = pic_gt.shape[0]
            
        y = meas            # [batch,256 256]
        Phi = mask.expand([batch_size1, 8, 256, 256])
        Phi_s = mask_s.expand([batch_size1, 256, 256])
        with torch.no_grad():
            out_pic_list = network(y, Phi, Phi_s)
            out_pic = out_pic_list[-1]

            print('>>>>>>len(out_pic_list)',len(out_pic_list))
            for s in range(stage_num):
                print('>>>>>>s, out_pic_list[s].shape', s, out_pic_list[s].shape) # [4,8,256,256]
                psnr_1 = 0
                for ii in range(meas.shape[0] * 8):
                    print('>>>>>>ii // 8, ii % 8,', ii // 8, ii % 8)
                    out_pic_p = out_pic_list[s][ii // 8, ii % 8, :, :]
                    gt_t = pic_gt[ii // 8, ii % 8, :, :]
                    rmse = torch.sqrt(criterion(out_pic_p, gt_t))
                    rmse = rmse.data
                    psnr_1 += 10 * torch.log10(1 / criterion(out_pic_p, gt_t))
                    print('computed')
                psnr_1 = psnr_1 / (meas.shape[0] * 8)
                psnr_stage[s, i] = psnr_1
        
        pred.append(out_pic.cpu().numpy())
        
    
    return pred, psnr_stage


def main():
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = './recon' + '/' + model_save_filename

    pred, psnr_stage = test(test_path1, result_path)
    psnr_stage_mean = torch.mean(psnr_stage, 1)
    for i in range(stage_num):    
        print("Stage {} psnr: {:.4f}".format(i, psnr_stage_mean[i]))
        

if __name__ == '__main__':
    main()
