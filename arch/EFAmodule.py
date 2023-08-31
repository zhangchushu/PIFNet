import torch 
import torch.nn as nn
import torch.nn.functional as F
from arch.IFEmodule import *
from myutils import *

class EFAS1_Atten(nn.Module):
    def __init__(self):
        super(EFAS1_Atten, self).__init__()

        self.aggre= nn.Sequential(

            nn.Conv3d(32, 32, (3,1,1), (2,1,1), (1,0,0)),
            nn.Conv3d(32, 32, (3,1,1), (1,1,1), (1,0,0)),
            nn.ReLU(True),
            # nn.Conv3d(32, 32, (12,1,1), (1,1,1), (1,0,0)),
            nn.Conv3d(32, 32, (12,1,1), 1, 0)
            )
    def forward(self, feas, events, e_ratio, img_ratio, mask ):
        L_fea, R_fea  = feas 
        L_voxel = e2v(L_fea, events, e_ratio, img_ratio, mask )
        R_voxel = e2v(R_fea, events, e_ratio, img_ratio, ~mask)
        
        L_flow = self.aggre(L_voxel).squeeze(2)
        R_flow = self.aggre(R_voxel).squeeze(2)
        

        return L_flow,R_flow
    

  
class EFAS2_Conv(nn.Module):
    def __init__(self):
        super(EFAS2_Conv, self).__init__()
        self.body1 = nn.Sequential(
            ResB(32*3),
            ResB(32*3),
            nn.Conv2d(32*3, 32*3, 3, 1, 1)
            )
        self.conv1 = nn.Conv2d(32*3, 32, 3, 1, 1)
        self.relu = nn.ReLU(True)


        self.body4 = nn.Sequential(
            ResB(32),
            ResB(32),
            nn.Conv2d(32,  32,  3, 1, 1)
            )
            
        self.conv3=   nn.Conv2d(32,  4,  3, 1, 1)


    def forward(self, x):
        x2 = self.conv1(self.relu(self.body1(x)  + x ))
        x3 =  self.conv3(self.relu(self.body4(x2)  + x2 ))

        return x3
