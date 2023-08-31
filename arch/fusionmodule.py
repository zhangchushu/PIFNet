import torch 
import torch.nn as nn
import torch.nn.functional as F
from arch.IFEmodule import *

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

  
        self.body1= nn.Sequential(          
            ResB(32*4+1),
            ResB(32*4+1),
            nn.Conv2d(32*4+1, 32*4+1,  3, 1, 1)
            )
        self.conv1= nn.Conv2d(32*4+1, 32*2,  3, 1, 1)

        self.body2= nn.Sequential(          
            ResB(32*2),
            ResB(32*2),
            nn.Conv2d(32*2, 32*2,  3, 1, 1)
            )
        self.conv2= nn.Conv2d(32*2, 32,  3, 1, 1)

        self.relu=  nn.ReLU(True)
        self.body3= nn.Sequential(    
            ResB(32),
            ResB(32),
            nn.Conv2d(32, 32,  3, 1, 1)
            )
        self.conv3= nn.Sequential(    
            ResB(32),
            nn.Conv2d(32, 32,  3, 1, 1)
            )

        self.body4 =nn.Sequential(
            ResB(32+6+1),
            ResB(32+6+1),
            nn.Conv2d(32+6+1, 32+6+1,  3, 1, 1),
        )
        self.conv4= nn.Conv2d(32+6+1, 3,  3, 1, 1)

    def forward(self,L_out,L_flow,img_flow,R_flow,R_out,L_img,R_img,stamp_pic):
        x = torch.cat([L_out,L_flow,img_flow,R_flow,R_out,stamp_pic],dim=1)
        x1 = self.conv1(self.relu(self.body1(x)+x))
        x2 = self.conv2(self.relu(self.body2(x1)+x1))
        x3 = self.conv3(self.relu(self.body3(x2)+x2))
        x4 = torch.cat([L_img,x3,R_img,stamp_pic],1)
        fusion_out = self.conv4(self.relu(self.body4(x4)+x4))

        return fusion_out