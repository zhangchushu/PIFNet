import os
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


class Goprosataset(Dataset):
    def __init__(self, data_root, is_training):
        self.data_root = data_root
        self.training = is_training
        if is_training:
            self.file_root = os.path.join(self.data_root, 'train')
        else:
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

    def _augmentation(self, img1, img2, img_gt, events,h,w):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        rotate = random.random() > 0.5

        if flip_h:
            img1 = img1[::-1, :, :]
            img2 = img2[::-1, :, :]
            img_gt = img_gt[::-1, :, :]
            events[:, 2] = (h-1)/h-events[:, 2]

        if flip_w:
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]
            img_gt = img_gt[:, ::-1, :]
            events[:, 1] = (w-1)/w-events[:, 1]

        if rotate:
            img1 = img1.transpose(1, 0, 2)
            img2 = img2.transpose(1, 0, 2)
            img_gt = img_gt.transpose(1, 0, 2)
            events = torch.cat([events[:, :1], events[:, 2:3], events[:, 1:2],events[:, 3:]], 1)

        return img1, img2, img_gt, events

    def __getitem__(self, index):
        if not self.training:
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
        h, w, _ = Im1.shape
        size = [160, 160]
        h_start = math.floor((h - size[0]) * random.random())
        h_end = h_start + size[0]
        w_start = math.floor((w - size[1]) * random.random())
        w_end = w_start + size[1]
        img1 = Im1[h_start:h_end, w_start:w_end, :]
        img2 = Im2[h_start:h_end, w_start:w_end, :]
        img_gt = Igt[h_start:h_end, w_start:w_end, :]

        # crop an event patch
        tp = (self.eventsdata[:, 0] > float(time1)) & (self.eventsdata[:, 0] < float(time2))
        eventpoints1 = self.eventsdata[tp, :]
        wp = (eventpoints1[:, 1] >= float(w_start)) & (eventpoints1[:, 1] < float(w_end))
        hp = (eventpoints1[:, 2] >= float(h_start)) & (eventpoints1[:, 2] < float(h_end))
        eventpoints = eventpoints1[wp & hp, :]
        # np.random.shuffle(eventpoints)
        # print(eventpoints.shape[0])

        alltime = eventpoints[:, 0:1]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 

        x = (eventpoints[:, 1:2] - w_start) / size[0]
        y = (eventpoints[:, 2:3] - h_start) /size[1]
        p = eventpoints[:, 3:] *2 -1

        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        eventpoints = torch.Tensor( eventpoints)


        num = 160*160*2
        len_neighbors  = [ num, num//4,  num//4//4,num//4//4//4,num//4//4, num//4, num]
        index_neighbors = [ num-1, num//4-1,  num//4//4-1,num//4//4//4-1,num//4//4//4-1, num//4//4-1, num//4-1]

        
        if self.training:

            img1, img2, img_gt, eventpoints = self._augmentation(img1, img2, img_gt, eventpoints,size[0],size[1])
            if eventpoints.shape[0] == 0:
                events = torch.zeros([num, 4])
                neighbors = findknn(events)

            elif eventpoints.shape[0] < num:  # 少了就在之后补零 (两边补零)
                n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
                eventpoints1 = eventpoints[n, :]

                neighbors = findknn(eventpoints1)
                num_in = eventpoints1.shape[0]
                eventpoints_aug = torch.zeros([   num  - num_in, 4])
                events = torch.vstack([eventpoints1, eventpoints_aug])
                
                for ki, l in enumerate( len_neighbors):
                    neighbor_aug = index_neighbors[ki] * torch.ones([  l - neighbors[ki].shape[0], neighbors[ki].shape[1] ],dtype=int)
                    neighbors[ki] =  torch.vstack([neighbors[ki], neighbor_aug])

            else:
                n = np.linspace(0,eventpoints.shape[0]-1,  num , dtype=int)
                events = eventpoints[n, :]
                neighbors = findknn(events)
       

        else:
            if eventpoints.shape[0] == 0:
                events = torch.zeros([num, 4])
                neighbors = findknn(events)

            elif eventpoints.shape[0] < num:  # 少了就在之后补零 (两边补零)
                n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
                eventpoints1 = eventpoints[n, :]

                neighbors = findknn(eventpoints1)
                num_in = eventpoints1.shape[0]
                eventpoints_aug = torch.zeros([   num  - num_in, 4])
                events = torch.vstack([eventpoints1, eventpoints_aug])
                
                for ki, l in enumerate( len_neighbors):
                    neighbor_aug = index_neighbors[ki] * torch.ones([  l - neighbors[ki].shape[0], neighbors[ki].shape[1] ],dtype=int)
                    neighbors[ki] =  torch.vstack([neighbors[ki], neighbor_aug])

            elif  eventpoints.shape[0] > num*2:
                n = np.linspace(0,eventpoints.shape[0]-1, num*2, dtype=int)
                events = eventpoints[n, :]
                neighbors = findknn(events)

            else: 
                n = np.linspace(0,eventpoints.shape[0]-1, eventpoints.shape[0]//64*64, dtype=int)
                events = eventpoints[n, :]
                neighbors = findknn(events)
           
        # events=eventpoints



        # augmentation
        # if self.training:
        #     img1, img2, img_gt, events = self._augmentation(img1, img2, img_gt, events)

        # to Tensor
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors

    def __len__(self):
        return self.filelength * 47

