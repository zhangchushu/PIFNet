import os
import time
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
from DataLoader.findknn import findknn
from nearest_neighbors import knn
from myutils import Plot

class Goprotestdataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.file_root = os.path.join(self.data_root, 'val')

        self.imageslist = []
        self.eventslist = []
        self.event_root = os.path.join(self.file_root, 'event')
        self.image_root = os.path.join(self.file_root, 'image_new')
        self.image_file = os.path.join(self.file_root, 'sharp_gray_lr')
        self.filelength = len(os.listdir(self.event_root))

        for npyfile in os.listdir(self.event_root):
            eventfiles = os.path.join(self.event_root, npyfile)
            self.eventslist.append(eventfiles)  # here

        for txtf in os.listdir(self.image_root):
            imgslines = []
            imgsfiles = os.path.join(self.image_root, txtf)
            with open(imgsfiles, 'r') as f_img:
                for line in f_img.readlines():
                    line = line.strip("\n").split()
                    imgslines.append(line)

            self.imageslist.append(imgslines)  # here
            self.eventsdata = []



    def __getitem__(self, index):
        # t1=time.time()
        random.seed(index)
        file_i = index // 47
        lines_i = index % 47
        time1 = self.imageslist[file_i][2*lines_i+2][0]
        time2 = self.imageslist[file_i][2*lines_i+4][0]
        image1 = self.imageslist[file_i][2*lines_i+2][1]
        image2 = self.imageslist[file_i][2*lines_i+4][1]
        imagegt = self.imageslist[file_i][2*lines_i+3][1]
        self.eventsdata = np.load(self.eventslist[file_i])

        img1path = os.path.join(self.image_file, image1)
        img2path = os.path.join(self.image_file, image2)
        img_gt_path = os.path.join(self.image_file, imagegt)
        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        # crop an image patch
        tp = (self.eventsdata[:, 0] > float(time1)) & (self.eventsdata[:, 0] < float(time2))
        eventpoints = self.eventsdata[tp, :]




        h, w, _ = Im1.shape
        the_size =max(h, w)

        if  h < the_size:
            img1 = np.concatenate([Im1,np.zeros((the_size-h,w,3))],0)
            img2 = np.concatenate([Im2,np.zeros((the_size-h,w,3))],0)
        elif w < the_size:
            img1 = np.concatenate([Im1,np.zeros((h,the_size-w,3))],1)
            img2 = np.concatenate([Im2,np.zeros((h,the_size-w,3))],1)
        else:
            img1=Im1
            img2=Im2

        img_gt =Igt



        alltime = eventpoints[:, 0:1]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 

        x = eventpoints[:, 1:2] /the_size
        y = eventpoints[:, 2:3] /the_size
        p = eventpoints[:, 3:] *2 -1

      
        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        eventpoints = torch.Tensor( eventpoints)

        # t1 = time.time()
        num = h * w
        len_neighbors  = [ num, num//4,  num//4//4,num//4//4//4,num//4//4, num//4, num]
        index_neighbors = [ num-1, num//4-1,  num//4//4-1,num//4//4//4-1,num//4//4//4-1, num//4//4-1, num//4-1]

        # t2=time.time()
        if eventpoints.shape[0] == 0:
            events = torch.zeros([4608, 4])
            neighbors = findknn(events)

        elif eventpoints.shape[0] < num :  # 少了就在之后补零 (两边补零)
            n = np.linspace(0,eventpoints.shape[0]-1,  eventpoints.shape[0]//64*64 , dtype=int)
            eventpoints1 = eventpoints[n, :]

            neighbors = findknn(eventpoints1 )
            num_in = eventpoints1.shape[0]
            eventpoints_aug = torch.zeros([  num   - num_in, 4])

            events = torch.vstack([eventpoints1, eventpoints_aug])
            

            for ki, l in enumerate( len_neighbors):
                neighbor_aug = index_neighbors[ki] * torch.ones([  l - neighbors[ki].shape[0], neighbors[ki].shape[1] ],dtype=int)
                neighbors[ki] =  torch.vstack([neighbors[ki], neighbor_aug])

        elif eventpoints.shape[0] > num * 4:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 4, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else :
            n = np.linspace(0,eventpoints.shape[0]-1,  eventpoints.shape[0]//64*64 , dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        # print("knn",time.time()-t2) 
        # else:
        #     events = torch.Tensor(eventpoints)
        #     neighbors = findknn(events)
    
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0
        # print('DataLoader:',time.time()-t1)
        return img1, img2, img_gt, events, neighbors,h,w

    def __len__(self):
        return self.filelength * 47

