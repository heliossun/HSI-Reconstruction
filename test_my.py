from dataLoadess import Imgdataset
from torch.utils.data import DataLoader
from models import ADMM_net_S9, ADMM_net_S12
from utils import generate_masks, time2file_name, A, At, gen_log
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from ssim_torch import ssim

from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

data_path = "/data/jiamianw/ICCV2021/DAVIS_data/training_data" # "./generated_data" #
test_path1 = "/data/jiamianw/BIRNAT-master/from_yangliu" # "./from_yangliu" #

last_train = 175
model_save_filename = '2021_12_20_02_56_04'
max_iter = 200
batch_size = 4
stage_num = 12


mask, mask_s = generate_masks(data_path)



dataset = Imgdataset(data_path)
train_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#first_frame_net = cnn1().cuda()

# if stage_num == 9:
#    network = ADMM_net_S9(n_resblocks = n_resblocks, n_feats = n_feats).cuda()
# elif stage_num == 12:
#    network = ADMM_net_S12(n_resblocks=n_resblocks, n_feats=n_feats).cuda()

if last_train != 0:
    network = torch.load(
        './model/' + model_save_filename + "/S{}_model_epoch_{}.pth".format(stage_num,last_train))

criterion  = nn.MSELoss()
criterion.cuda()





def torch_ssim(img, ref):
    img_add = torch.unsqueeze(img,0)
    ref_add = torch.unsqueeze(ref,0)
    return ssim(torch.unsqueeze(img_add,0), torch.unsqueeze(ref_add,0))

def test(test_path, epoch, result_path, psnr_epoch, ssim_epoch):
    test_list = os.listdir(test_path)
    psnr_sample = torch.zeros(len(test_list))
    ssim_sample = torch.zeros(len(test_list))
    print('name=', test_list)
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

            # print('>>>>>>out_pic, type, shape=',type(out_pic), out_pic.shape) # tensor [4,8,256,256]
            # print('>>>>>>pic_gt, type, shape=',type(pic_gt), pic_gt.shape) # tensor [4,8,256,256]

            psnr_1 = 0
            ssim_1 = 0
            for ii in range(meas.shape[0] * 8):
                out_pic_p = out_pic[ii // 8, ii % 8, :, :]
                gt_t = pic_gt[ii // 8, ii % 8, :, :]

                # print('>>>>>>out_pic_p, type, shape=',type(out_pic_p), out_pic_p.shape) # tensor [256,256]
                # print('>>>>>>gt_t, type, shape=',type(gt_t), gt_t.shape) # tensor [256,256]

                rmse = torch.sqrt(criterion(out_pic_p, gt_t))
                rmse = rmse.data
                psnr_1 += 10 * torch.log10(1 / criterion(out_pic_p, gt_t))
                ssim_1 += torch_ssim(out_pic_p, gt_t)
            ssim_1 = ssim_1 / (meas.shape[0] * 8)
            ssim_sample[i] = ssim_1
            psnr_1 = psnr_1 / (meas.shape[0] * 8)
            psnr_sample[i] = psnr_1
        
        pred.append(out_pic.cpu().numpy())
        
    psnr_epoch.append(psnr_sample)
    ssim_epoch.append(ssim_sample)

    return pred, psnr_epoch, ssim_epoch




    


def main():
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    psnr_epoch = []
    psnr_max = 0
    ssim_epoch = []

    print('----------Train: epoch%d----------'%last_train)
    pred, psnr_epoch, ssim_epoch = test(test_path1, last_train, result_path, psnr_epoch, ssim_epoch)
    print('psnr_epoch=', psnr_epoch)
    print('ssim_epoch=', ssim_epoch)

    psnr_mean = torch.mean(psnr_epoch[-1])
    ssim_mean = torch.mean(ssim_epoch[-1])

    print("Test PSNR result: {:.4f}".format(psnr_mean))
    print("Test SSIM result: {:.4f}".format(ssim_mean))

    scio.savemat('./ours.mat',{'pred':pred})


if __name__ == '__main__':
    main()
