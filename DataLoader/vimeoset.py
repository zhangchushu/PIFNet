import os
from sqlite3 import Timestamp
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

from myutils import *

class vimeotestset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'events')
        self.image_root = os.path.join(self.data_root, 'images')


        self.imageslist = []
        self.eventslist = []
        for file in os.listdir(self.event_root):
            e_path = os.path.join(self.event_root, file)        #  event/00001
            im_path = os.path.join(self.image_root, file )
            for subfile in   os.listdir(e_path):
                im_flie =  os.path.join(im_path, subfile)
                e_flie =  os.path.join(e_path, subfile)
                self.eventslist.append(e_flie)  # here
                self.imageslist.append(im_flie)  # here

        self.filelength = len(self.imageslist)    #00001  *100

    def __getitem__(self, index):

        image_path = self.imageslist[index]
        events_path = self.eventslist[index]

        img1path = os.path.join(image_path, 'imgs', 'im1.png')
        img2path = os.path.join(image_path, 'imgs', 'im3.png')
        img_gt_path = os.path.join(image_path, 'imgs', 'im2.png')

        events1_path = os.path.join(events_path, 'events0.npy')
        events2_path = os.path.join(events_path, 'events1.npy')

        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)
        Im1 = cv.cvtColor(Im1, cv.COLOR_BGR2RGB)
        Im2 = cv.cvtColor(Im2, cv.COLOR_BGR2RGB)
        Igt = cv.cvtColor(Igt, cv.COLOR_BGR2RGB)

        events1 = np.load(events1_path)
        events2 = np.load(events2_path)

        eventpoints = np.concatenate([events1, events2], 1).transpose(1, 0)

        h, w, _ = Im1.shape
        the_size = max(h, w)

        if h < the_size:
            img1 = np.concatenate([Im1, np.zeros((the_size - h, w, 3))], 0)
            img2 = np.concatenate([Im2, np.zeros((the_size - h, w, 3))], 0)
        elif w < the_size:
            img1 = np.concatenate([Im1, np.zeros((h, the_size - w, 3))], 1)
            img2 = np.concatenate([Im2, np.zeros((h, the_size - w, 3))], 1)
        else:
            img1 = Im1
            img2 = Im2

        img_gt = Igt

        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints[:, 0:1] / the_size
        y = eventpoints[:, 1:2] / the_size
        p = eventpoints[:, 3:]

        eventpoints = np.concatenate([alltime, x, y, p], 1)
        eventpoints = torch.Tensor(eventpoints)

        num= h*w//64*64

        len_neighbors = [num, num // 4, num // 4 // 4, num // 4 // 4 // 4, num // 4 // 4, num // 4, num]
        index_neighbors = [num - 1, num // 4 - 1, num // 4 // 4 - 1, num // 4 // 4 // 4 - 1, num // 4 // 4 // 4 - 1,
                           num // 4 // 4 - 1, num // 4 - 1]

        num_neighbors = [16, 16, 16, 16, 1, 1, 1]

        if eventpoints.shape[0] == 0:
            events = torch.zeros([4608, 4])
            neighbors = []
            for ki, l in enumerate(len_neighbors):
                neighbors.append(torch.zeros([l, num_neighbors[ki]], dtype=torch.int64))

        elif eventpoints.shape[0] < num:  # 少了就在之后补零 (两边补零)
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            eventpoints1 = eventpoints[n, :]

            neighbors = findknn(eventpoints1)
            num_in = eventpoints1.shape[0]
            eventpoints_aug = torch.zeros([num - num_in, 4])

            events = torch.vstack([eventpoints1, eventpoints_aug])

            for ki, l in enumerate(len_neighbors):
                neighbor_aug = index_neighbors[ki] * torch.ones([l - neighbors[ki].shape[0], neighbors[ki].shape[1]],
                                                                dtype=int)
                neighbors[ki] = torch.vstack([neighbors[ki], neighbor_aug])

        elif eventpoints.shape[0] > num * 4:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 4, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        # else:
        #     events = torch.Tensor(eventpoints)
        #     neighbors = findknn(events)
        # print(time.time()-t1)
        Timestamp = 0.5
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors, h, w, Timestamp

    def __len__(self):
        return len(self.eventslist)







class vimeotrainset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'events256')
        self.image_root = os.path.join(self.data_root, 'images')
        self.imageslist = []
        self.eventslist = []
        for file in os.listdir(self.event_root):
            e_path = os.path.join(self.event_root, file)  # event/00001
            im_path = os.path.join(self.image_root, file)
            for subfile in os.listdir(e_path):
                im_flie = os.path.join(im_path, subfile)
                e_flie = os.path.join(e_path, subfile)

                self.eventslist.append(e_flie)  # here
                self.imageslist.append(im_flie)  # here

        self.filelength = len(self.imageslist)  # 00001  *10

    def _augmentation(self, img1, img2, img_gt, events):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        rotate = random.random() > 0.5
        h, w, c = img1.shape
        if flip_h:
            img1 = img1[::-1, :, :]
            img2 = img2[::-1, :, :]
            img_gt = img_gt[::-1, :, :]
            events[:, 2] = (h - 1) / h - events[:, 2]

        if flip_w:
            img1 = img1[:, ::-1, :]
            img2 = img2[:, ::-1, :]
            img_gt = img_gt[:, ::-1, :]
            events[:, 1] = (w - 1) / w - events[:, 1]

        if rotate:
            img1 = img1.transpose(1, 0, 2)
            img2 = img2.transpose(1, 0, 2)
            img_gt = img_gt.transpose(1, 0, 2)
            events = np.concatenate([events[:, :1], events[:, 2:3], events[:, 1:2], events[:, 3:]], 1)

        return img1, img2, img_gt, events

    def __getitem__(self, index):
        random.seed(index)

        image_path = self.imageslist[index]
        events_path = self.eventslist[index]

        ##随机抽取三张图片
        piclist = random.sample([2, 3, 4, 5, 6], 1)

        img1path = os.path.join(image_path, 'imgs', 'im1.png')
        img2path = os.path.join(image_path, 'imgs', 'im7.png')
        img_gt_path = os.path.join(image_path, 'imgs', 'im' + str(piclist[0]) + '.png')

        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        h, w, _ = Im1.shape
        size = [224, 224]

        h_start = math.floor((256 - size[0]) * random.random())
        h_end = h_start + size[0]
        w_start = math.floor((256 - size[1]) * random.random()) + 96
        w_end = w_start + size[1]
        img1 = Im1[h_start:h_end, w_start:w_end, :]
        img2 = Im2[h_start:h_end, w_start:w_end, :]
        img_gt = Igt[h_start:h_end, w_start:w_end, :]

        # t1=time.time()
        neighbors = []

        for efile in os.listdir(events_path):
            eventpoints = np.load(os.path.join(events_path, efile))
        eventpoints[:, 1] = eventpoints[:, 1] * 256 + 96
        eventpoints[:, 2] = eventpoints[:, 2] * 256

        wp = (eventpoints[:, 1] >= float(w_start)) & (eventpoints[:, 1] < float(w_end))
        hp = (eventpoints[:, 2] >= float(h_start)) & (eventpoints[:, 2] < float(h_end))
        eventpoints = eventpoints[wp & hp, :]

        eventpoints[:, 1] = (eventpoints[:, 1] - w_start) / size[0]
        eventpoints[:, 2] = (eventpoints[:, 2] - h_start) / size[1]

        img1, img2, img_gt, eventpoints = self._augmentation(img1, img2, img_gt, eventpoints)
        # checkdisplay2(img1,img2,eventpoints,is_save=False,is_show=True)

        timestamp = (piclist[0] - 1) / 6

        eventpoints = torch.Tensor(eventpoints)

        num = size[0] * size[1] * 2
        len_neighbors = [num, num // 4, num // 4 // 4, num // 4 // 4 // 4, num // 4 // 4, num // 4, num]
        index_neighbors = [num - 1, num // 4 - 1, num // 4 // 4 - 1, num // 4 // 4 // 4 - 1, num // 4 // 4 // 4 - 1,
                           num // 4 // 4 - 1, num // 4 - 1]

        num_neighbors = [16, 16, 16, 16, 1, 1, 1]

        if eventpoints.shape[0] == 0:
            events = torch.zeros([num, 4])
            neighbors = []
            for ki, l in enumerate(len_neighbors):
                neighbors.append(torch.zeros([l, num_neighbors[ki]], dtype=torch.int64))


        elif eventpoints.shape[0] < num:  # 少了就在之后补零 (两边补零)
            neighbors = findknn(eventpoints)
            num_in = eventpoints.shape[0]
            eventpoints_aug = torch.zeros([num - num_in, 4])
            events = torch.vstack([eventpoints, eventpoints_aug])

            for ki, l in enumerate(len_neighbors):
                neighbor_aug = index_neighbors[ki] * torch.ones([l - neighbors[ki].shape[0], neighbors[ki].shape[1]],
                                                                dtype=int)
                neighbors[ki] = torch.vstack([neighbors[ki], neighbor_aug])
            # print("1",time.time()-ti)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, num, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)

        # print("knn2",time.time()-t1)
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors, timestamp

    def __len__(self):
        return len(self.eventslist)

