from inspect import FrameInfo
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
from myutils import *
from DataLoader.findknn import findknn
from nearest_neighbors import knn




class HSBSRGB_tripple(Dataset):
    def __init__(self, data_root):

        self.data_root = data_root

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []

        for subfile in os.listdir(self.data_root):

            self.event_root = os.path.join(self.data_root, subfile, 'events')
            self.image_root = os.path.join(self.data_root, subfile, 'images')

            pic_num = len(os.listdir(self.event_root))

            for i in range(4, pic_num, 4):
                img1 = str(i - 4).zfill(6) + ".png"
                img2 = str(i).zfill(6) + ".png"

                events1 = str(i - 4).zfill(6) + ".npz"
                events2 = str(i - 3).zfill(6) + ".npz"
                events3 = str(i - 2).zfill(6) + ".npz"
                events4 = str(i - 1).zfill(6) + ".npz"

                e_flie = [os.path.join(self.event_root, events1), os.path.join(self.event_root, events2),
                          os.path.join(self.event_root, events3), os.path.join(self.event_root, events4)]

                img1path = os.path.join(self.image_root, img1)
                img2path = os.path.join(self.image_root, img2)
                for index in range(3):
                    k = 3 - index
                    img_gt = str(i - k).zfill(6) + ".png"

                    img_gt_path = os.path.join(self.image_root, img_gt)

                    self.eventslist.append(e_flie)  # here

                    self.image1list.append(img1path)  # here
                    self.image2list.append(img2path)  # here
                    self.imagegtlist.append(img_gt_path)  # here
                    self.flag.append(index)

        self.filelength = len(self.image1list)  # 00001  *100

    def __getitem__(self, index):

        events_path = self.eventslist[index]

        img1path = self.image1list[index]
        img2path = self.image2list[index]
        img_gt_path = self.imagegtlist[index]
        flag = self.flag[index]

        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        Im1 = cv.cvtColor(Im1, cv.COLOR_BGR2RGB)
        Im2 = cv.cvtColor(Im2, cv.COLOR_BGR2RGB)
        Igt = cv.cvtColor(Igt, cv.COLOR_BGR2RGB)


        h, w, _ = Im1.shape
        the_size = max(h, w)


        img1 = Im1
        img2 = Im2
        img_gt = Igt


        eventpoints = []
        for e_path in events_path:
            e_file = np.load(e_path, allow_pickle=True)
            x = e_file['x'].reshape(-1, 1) / 32
            y = e_file['y'].reshape(-1, 1) / 32
            alltime = e_file['timestamp'].reshape(-1, 1).astype(np.float64)
            p = e_file['polarity'].reshape(-1, 1).astype(np.float64) * 2 - 1

            txyp = np.concatenate([alltime, x, y, p], 1)
            eventpoints.append(torch.Tensor(txyp))

        eventpoints = torch.vstack(eventpoints)

        wp = eventpoints[:, 1] < float(w)
        hp = eventpoints[:, 2] < float(h)
        eventpoints1 = eventpoints[wp & hp, :]

        alltime = eventpoints1[:, 0:1]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints1[:, 1:2] / the_size
        y = eventpoints1[:, 2:3] / the_size
        p = eventpoints1[:, 3:]

        eventpoints = torch.hstack((alltime, x, y, p))

        # checkdisplay2(img1,img2,eventpoints ,is_show=True)
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

        elif eventpoints.shape[0] > num * 3:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 3, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        # checkdisplay2(img1,img2,events,is_show=True)

        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        Timestamp = (flag+1)/4

        return img1, img2, img_gt,  events, neighbors, h, w, Timestamp

    def __len__(self):
        return self.filelength



class HSBSRGB_single(Dataset):
    def __init__(self, data_root):

        self.data_root = data_root

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []

        for subfile in os.listdir(self.data_root):

            self.event_root = os.path.join(self.data_root, subfile, 'events')
            self.image_root = os.path.join(self.data_root, subfile, 'images')

            pic_num = len(os.listdir(self.event_root))

            for i in range(2, pic_num, 2):
                img1 = str(i - 2).zfill(6) + ".png"
                img2 = str(i).zfill(6) + ".png"
                img_gt = str(i -1).zfill(6) + ".png"

                events1 = str(i - 2).zfill(6) + ".npz"
                events2 = str(i - 1).zfill(6) + ".npz"

                e_flie = [os.path.join(self.event_root, events1), os.path.join(self.event_root, events2)]

                img1path = os.path.join(self.image_root, img1)
                img2path = os.path.join(self.image_root, img2)
                img_gt_path = os.path.join(self.image_root, img_gt)

                self.eventslist.append(e_flie)  # here

                self.image1list.append(img1path)  # here
                self.image2list.append(img2path)  # here
                self.imagegtlist.append(img_gt_path)  # here


        self.filelength = len(self.image1list)  # 00001  *100

    def __getitem__(self, index):

        events_path = self.eventslist[index]

        img1path = self.image1list[index]
        img2path = self.image2list[index]
        img_gt_path = self.imagegtlist[index]


        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)

        Im1 = cv.cvtColor(Im1, cv.COLOR_BGR2RGB)
        Im2 = cv.cvtColor(Im2, cv.COLOR_BGR2RGB)
        Igt = cv.cvtColor(Igt, cv.COLOR_BGR2RGB)


        h, w, _ = Im1.shape
        the_size = max(h, w)

        img1 = Im1
        img2 = Im2

        img_gt = Igt


        eventpoints = []
        for e_path in events_path:
            e_file = np.load(e_path, allow_pickle=True)
            x = e_file['x'].reshape(-1, 1) / 32
            y = e_file['y'].reshape(-1, 1) / 32
            alltime = e_file['timestamp'].reshape(-1, 1).astype(np.float64)
            p = e_file['polarity'].reshape(-1, 1).astype(np.float64) * 2 - 1

            txyp = np.concatenate([alltime, x, y, p], 1)
            eventpoints.append(torch.Tensor(txyp))

        eventpoints = torch.vstack(eventpoints)

        wp = eventpoints[:, 1] < float(w)
        hp = eventpoints[:, 2] < float(h)
        eventpoints1 = eventpoints[wp & hp, :]

        alltime = eventpoints1[:, 0:1]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints1[:, 1:2] / the_size
        y = eventpoints1[:, 2:3] / the_size
        p = eventpoints1[:, 3:]

        eventpoints = torch.hstack((alltime, x, y, p))

        # checkdisplay2(img1,img2,eventpoints ,is_show=True)
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

        elif eventpoints.shape[0] > num * 3:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 3, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        # checkdisplay2(img1,img2,events,is_show=True)

        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        Timestamp = 0.5

        return img1, img2, img_gt,  events, neighbors, h, w, Timestamp

    def __len__(self):
        return self.filelength
