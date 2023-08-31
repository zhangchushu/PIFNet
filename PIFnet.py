from mimetypes import init
from tkinter import N
from turtle import left
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from cv2 import resizeWindow, transpose
from numpy import reshape
from arch.EFEmodule import EFEmodule
from arch.IFEmodule import *
from arch.fusionmodule import FusionModule
from arch.EFAmodule import EFAS1_Atten,EFAS2_Conv
from myutils import *
import pylab




"""
-----------------------主网络部分-------------------------------------------------------
"""
class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.EFE = EFEmodule(1, n_neighbors=16)
        self.IFE = IFEmodule()

        self.EFAS1 = EFAS1_Atten()
        self.EFAS2 = EFAS2_Conv()

        self.FM = FusionModule()

    def forward(self, events, img1, img2, neighbors, t_stamplist):
        b, _, h, w = img1.shape

        imgfeatures = self.IFE(img1,img2)    #可以尝试不同的光流网络
        img_flow, img_ratio,img1_fea,img2_fea =imgfeatures
        

        eventfeatures = self.EFE([events, neighbors ])
        evn_fea, e_ratio= eventfeatures

        imgs = []
        
        for t_stamp in t_stamplist:

            L_fea, R_fea, mask = events_split(evn_fea, events, t_stamp)
            feas = L_fea, R_fea

            
            L_flow,R_flow =self.EFAS1(feas, events, e_ratio, img_ratio, mask )
            flow = self.EFAS2(torch.cat([L_flow, img_flow, R_flow], dim=1))

            flow1 = flow[:,:2,...]
            flow2 = flow[:,2:,...]

            L_out =  optical_flow_warp(img1_fea, flow1)
            R_out =  optical_flow_warp(img2_fea, flow2)


            L_img =  optical_flow_warp(img1, flow1)
            R_img =  optical_flow_warp(img2, flow2)


            stamp_pic= t_stamp.view(-1, 1, 1, 1).expand(b, 1, h, w).to(img1.device)
            fusion_out = self.FM(L_out,L_flow,img_flow,R_flow,R_out,L_img,R_img,stamp_pic)

            
            imgs.append((fusion_out))

        return imgs, flow1, flow2







