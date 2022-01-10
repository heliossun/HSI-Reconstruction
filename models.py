from utils import A, At
from SRN_model import SRN
import torch
import torch.nn as nn
from Model import build_model
#替换SRn为 swin transformer, 看一下输入的shape什么的，参数量，运算时间，看一下有没有针对transformer需要做的优化，optimizer之类的

class ADMM_net_S4(nn.Module):

    def __init__(self, n_resblocks = 8, n_feats = 16, cfg=None):
        super(ADMM_net_S4, self).__init__()
        self.Swin_T = build_model(cfg)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s):
        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-4
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        

        output_list = x_list[-3:]
        return output_list
    
class ADMM_net_S9(nn.Module):

    def __init__(self, n_resblocks = 8, n_feats = 16, cfg=None):
        super(ADMM_net_S9, self).__init__()
        self.Swin_T = build_model(cfg)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s):
        x_list = []
        theta = At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma8),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = A(theta+b,Phi)
        x = theta+b + At(torch.div(y-yb,Phi_s+self.gamma9),Phi)
        x1 = x-b
        theta = self.Swin_T(x1)
        b = b- (x-theta)
        x_list.append(theta)

        output_list = x_list[-3:]
        return output_list


class ADMM_net_S12(nn.Module):

    def __init__(self, n_resblocks=8, n_feats=16,cfg=None):
        super(ADMM_net_S12, self).__init__()

        self.unet1 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet2 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet3 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet4 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet5 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet6 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet7 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet8 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet9 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet10 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet11 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.unet12 = SRN(n_resblocks, n_feats, 8, 8)  # Unet(8, 8)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma10 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma11 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma12 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s):
        x_list = []
        theta = At(y, Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma1), Phi)
        x1 = x - b
        theta = self.unet1(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma2), Phi)
        x1 = x - b
        theta = self.unet2(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma3), Phi)
        x1 = x - b
        theta = self.unet3(x1)
        b = b - (x - theta)
        x_list.append(theta)
        ### 4-6
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma4), Phi)
        x1 = x - b
        theta = self.unet4(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma5), Phi)
        x1 = x - b
        theta = self.unet5(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma6), Phi)
        x1 = x - b
        theta = self.unet6(x1)
        b = b - (x - theta)
        x_list.append(theta)
        ### 7-9
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma7), Phi)
        x1 = x - b
        theta = self.unet7(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma8), Phi)
        x1 = x - b
        theta = self.unet8(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma9), Phi)
        x1 = x - b
        theta = self.unet9(x1)
        b = b - (x - theta)
        x_list.append(theta)
        ### 10-12
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma10), Phi)
        x1 = x - b
        theta = self.unet10(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma11), Phi)
        x1 = x - b
        theta = self.unet11(x1)
        b = b - (x - theta)
        x_list.append(theta)
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma12), Phi)
        x1 = x - b
        theta = self.unet12(x1)
        b = b - (x - theta)
        x_list.append(theta)

        output_list = x_list[-3:]
        return output_list
