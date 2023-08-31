from asyncio import events
import math
from tkinter import E
from turtle import forward
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils import *
from torchvision import transforms


class ResB(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.body(x) + x)


class IFEmodule(nn.Module):
    def __init__(self):
        super(IFEmodule,self).__init__()
        self._size_adapter = SizeAdapter(minimum_size=16)
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            ResB(16)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(32)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(48)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            ResB(64)
        )

        self.dconv1 =   nn.Sequential(
            ResB(128),
            ResB(128),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.relu = nn.ReLU(True)
        self.dconv2 =nn.Conv2d(128, 128, 3, 1, 1)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128+96, 96, 3, 1, 1),
            nn.ReLU(True),
            ResB(96)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(96+64, 64, 3, 1, 1),
            nn.ReLU(True),
            ResB(64)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(64+32, 32, 3, 1, 1),
            nn.ReLU(True),
            ResB(32)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.tail = nn.Sequential(
            ResB(32),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        
    def forward(self, image1,image2):
        image1 = self._size_adapter.pad(image1)
        image2 = self._size_adapter.pad(image2)
        fea1_1 = self.encoder1(image1)   #16
        fea2_1 = self.encoder1(image2)

        fea1_2 = self.encoder2(fea1_1)   #32
        fea2_2 = self.encoder2(fea2_1)

        fea1_3 = self.encoder3(fea1_2)  #48
        fea2_3 = self.encoder3(fea2_2)
 
        fea1_4 = self.encoder4(fea1_3)  #64
        fea2_4 = self.encoder4(fea2_3)

        x = torch.cat([fea1_4,fea2_4],1)
        fea4 = self.dconv2(self.relu(self.dconv1(x)+x) ) #128
        # fea4 = self.dconv1(torch.cat([fea1_4,fea2_4],1))  #128
        fea5 = self.decoder1(torch.cat([self.upsample(fea4), fea1_3, fea2_3], 1))
        fea6 = self.decoder2(torch.cat([self.upsample(fea5), fea1_2, fea2_2], 1))
        fea7 = self.decoder3(torch.cat([self.upsample(fea6), fea1_1, fea2_1], 1))


        im_ratio = self.tail(fea7)


        return self._size_adapter.unpad(fea7), self._size_adapter.unpad(im_ratio),self._size_adapter.unpad(fea1_1),self._size_adapter.unpad(fea2_1)
