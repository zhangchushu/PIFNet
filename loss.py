# Sourced from https://github.com/myungsub/CAIN/blob/master/loss.py, who sourced from https://github.com/thstkdgus35/EDSR-PyTorch/tree/master/src/loss
# Added Huber loss in addition.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import pytorch_msssim
from myutils import optical_flow_warp


import torch.optim as optim

vgg16=models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3 .cuda()
for param in vgg16_conv_4_3.parameters():
    param.requires_grad = False
    
def flow_smooth_loss(flow):
    loss = (flow[:, :, :-1, :-1] - flow[:, :, 1:, :-1]).abs() + (flow[:, :, :-1, :-1] - flow[:, :, :-1, 1:]).abs()

    return loss.mean()

# Wrapper of loss functions
class Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        self.loss = nn.L1Loss()
        self.mse =nn.MSELoss()
    def forward(self, sr, hr, img1, img2, flow1, flow2):
        loss_sr = self.loss(sr, hr)
        loss_prep = self.mse(vgg16_conv_4_3(sr), vgg16_conv_4_3(hr))
        loss_flow1 = self.loss(optical_flow_warp(img1, flow1), hr) + 0.1 * flow_smooth_loss(flow1)
        loss_flow2 = self.loss(optical_flow_warp(img2, flow2), hr) + 0.1 * flow_smooth_loss(flow2)

        return loss_sr, 0.5 * (loss_flow1 + loss_flow2),loss_prep
