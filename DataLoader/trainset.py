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


class vimeotestset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'event')
        self.image_root = os.path.join(self.data_root, 'image_new')

        self.filelength = len(os.listdir(self.event_root))    #00001  *100

        self.imageslist = []
        self.eventslist = []
        for event_file in os.listdir(self.event_root):
            e_path = os.path.join(self.event_root, event_file)        #  event/00001   
            for e_subfile in   os.listdir(e_path):
                e_flie =  os.path.join(e_path, e_subfile) 
                self.eventslist.append(e_flie)  # here


        for image_file in os.listdir(self.image_root):
            im_path = os.path.join(self.image_root, image_file)        #  event/00001   
            for im_subfile in   os.listdir(im_path):
                im_flie =  os.path.join(im_path, im_subfile) 
                self.imageslist.append(im_flie)  # here
    def _augmentation(self, img1, img2, img_gt, events):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        rotate = random.random() > 0.5

        if flip_h:
            img1 = img1[::-1, :, :]
            img2 = img2[::-1, :, :]
            img_gt = img_gt[::-1, :, :]
            events[:, 2][events[:, 2]>0] = 1-events[:, 2][events[:, 2]>0]

        if flip_w:
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]
            img_gt = img_gt[:, ::-1, :]
            events[:, 1][events[:, 1]>0] = 1-events[:, 1][events[:, 1]>0]

        if rotate:
            img1 = img1.transpose(1, 0, 2)
            img2 = img2.transpose(1, 0, 2)
            img_gt = img_gt.transpose(1, 0, 2)
            events = torch.cat([events[:, :1], events[:, 2:3], events[:, 1:2],events[:, 3:]], 1)

        return img1, img2, img_gt, events



    def __getitem__(self, index):

        image_path  = self.imageslist[index]
        events_path = self.eventslist[index]

        img1path = os.path.join(image_path, 'im0')
        img2path = os.path.join(image_path, 'im2')
        img_gt_path = os.path.join(image_path, 'im1')

        events_path = os.path.join(events_path, 'events0')
     
        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        eventpoints = np.load(events_path)  

        # crop an image patch
        h, w, _ = Im1.shape
        the_size =196

        h_start = math.floor((h - the_size) * random.random())
        h_end = h_start + the_size
        w_start = math.floor((w - the_size) * random.random())
        w_end = w_start + the_size
        img1 = Im1[h_start:h_end, w_start:w_end, :]
        img2 = Im2[h_start:h_end, w_start:w_end, :]
        img_gt = Igt[h_start:h_end, w_start:w_end, :]




        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) * (the_size -1)

        x = eventpoints[:, 0:1] 
        y = eventpoints[:, 1:2] 
        p = eventpoints[:, 3:] 

        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        txy = eventpoints [:, :3]


        if eventpoints.shape[0] == 0:
            events = np.zeros([4608, 4])
        elif eventpoints.shape[0] < 4608:  # 少了就在之后补零 (两边补零)
            num_in = eventpoints.shape[0]
            eventpoints_aug = np.zeros([4608 - num_in, 4])
            events = np.concatenate([eventpoints, eventpoints_aug], 0)
        elif eventpoints.shape[0] > 46080:
            n = np.linspace(0,eventpoints.shape[0]-1,4608, dtype=int)
            events = eventpoints[n, :]
        else:
            events = eventpoints
        np.random.shuffle(events)

        events = torch.Tensor(events)
        events = torch.cat([events[:, :3] / (the_size-1), events[:, 3:]], 1).float()
        neighbors = findknn(events)


        img1, img2, img_gt, events = self._augmentation(img1, img2, img_gt, events)
        # to Tensor
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors,h,w

    def __len__(self):
        return len(self.eventslist)

