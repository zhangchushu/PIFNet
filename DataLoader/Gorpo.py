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


class Gopro7(Dataset):
    def __init__(self):

        self.event_root = r"/media/root/LENOVO_USB_HDD/GoPro/test_events"
        self.image_root = r"/media/root/LENOVO_USB_HDD/GoPro/test_imgs"
        # self.event_root = r"F:\2022_7_11\GOPRO\train_events"
        # self.image_root = r"F:\2022_7_11\GOPRO\GOPRO_Large_all\train_images"

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []

        for subfile in os.listdir(self.event_root):
            # if subfile != "GOPR0410_11_00":
            #     continue
            event_root = os.path.join(self.event_root, subfile)
            image_root = os.path.join(self.image_root, subfile, "imgs")
            img_list = os.listdir(image_root)
            eve_list = os.listdir(event_root)
            eve_list.sort()
            img_list.sort()
            pic_num = len(img_list)

            for i in range(8, pic_num, 8):
                img1 = img_list[i - 8]
                img2 = img_list[i]
                img1path = os.path.join(image_root, img1)
                img2path = os.path.join(image_root, img2)
                events1 = eve_list[i - 8]
                events2 = eve_list[i - 7]
                events3 = eve_list[i - 6]
                events4 = eve_list[i - 5]
                events5 = eve_list[i - 4]
                events6 = eve_list[i - 3]
                events7 = eve_list[i - 2]
                events8 = eve_list[i - 1]
                e_flie = [os.path.join(event_root, events1), os.path.join(event_root, events2),
                          os.path.join(event_root, events3), os.path.join(event_root, events4),
                          os.path.join(event_root, events5), os.path.join(event_root, events6),
                          os.path.join(event_root, events7), os.path.join(event_root, events8)]
                for index in range(7):
                    k = 7 - index
                    img_gt = img_list[i - k]
                    img_gt_path = os.path.join(image_root, img_gt)

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

        timestamp = (flag + 1) / 8
        eventpoints = []
        for file in events_path:
            eventpoints.append(torch.Tensor(np.load(file)))
        eventpoints = torch.vstack(eventpoints)

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

        # checkdisplay_withimg(eventpoints,img1 )
        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints[:, 0:1] / the_size
        y = (eventpoints[:, 1:2] + 8) / the_size
        p = eventpoints[:, 3:]
        eventpoints = np.concatenate([alltime, x, y, p], 1)
        eventpoints = torch.Tensor(eventpoints)

        # checkdisplay2(img1,img2,eventpoints ,is_show=True)
        num = h * w // 64 * 64
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

        elif eventpoints.shape[0] > num :
            n = np.linspace(0, eventpoints.shape[0] - 1, num , dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors, h, w, timestamp

    def __len__(self):
        return len(self.eventslist)


class Gopro3(Dataset):
    def __init__(self):

        self.event_root = r"/media/root/LENOVO_USB_HDD/GoPro/test_events"
        self.image_root = r"/media/root/LENOVO_USB_HDD/GoPro/test_imgs"
        # self.event_root = r"F:\2022_7_11\GOPRO\train_events"
        # self.image_root = r"F:\2022_7_11\GOPRO\GOPRO_Large_all\train_images"

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []

        for subfile in os.listdir(self.event_root):
            # if subfile != "GOPR0410_11_00":
            #     continue
            event_root = os.path.join(self.event_root, subfile)
            image_root = os.path.join(self.image_root, subfile, "imgs")
            img_list = os.listdir(image_root)
            eve_list = os.listdir(event_root)
            eve_list.sort()
            img_list.sort()
            pic_num = len(img_list)

            for i in range(4, pic_num, 4):
                img1 = img_list[i - 4]
                img2 = img_list[i]
                img1path = os.path.join(image_root, img1)
                img2path = os.path.join(image_root, img2)
                events1 = eve_list[i - 4]
                events2 = eve_list[i - 3]
                events3 = eve_list[i - 2]
                events4 = eve_list[i - 1]
                e_flie = [os.path.join(event_root, events1), os.path.join(event_root, events2),
                          os.path.join(event_root, events3), os.path.join(event_root, events4)]
                for index in range(3):
                    k = 3 - index
                    img_gt = img_list[i - k]
                    img_gt_path = os.path.join(image_root, img_gt)

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

        timestamp = (flag + 1) / 4
        eventpoints = []
        for file in events_path:
            eventpoints.append(torch.Tensor(np.load(file)))
        eventpoints = torch.vstack(eventpoints)

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

        # checkdisplay_withimg(eventpoints,img1 )
        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints[:, 0:1] / the_size
        y = (eventpoints[:, 1:2] + 8) / the_size
        p = eventpoints[:, 3:]
        eventpoints = np.concatenate([alltime, x, y, p], 1)
        eventpoints = torch.Tensor(eventpoints)

        # checkdisplay2(img1,img2,eventpoints ,is_show=True)
        num = h * w // 64 * 64
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

        elif eventpoints.shape[0] > num :
            n = np.linspace(0, eventpoints.shape[0] - 1, num , dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors, h, w, timestamp

    def __len__(self):
        return len(self.eventslist)


class Gopro15(Dataset):
    def __init__(self):

        self.event_root = r"F:\2022_7_11\GOPRO\train_events"
        self.image_root = r"F:\2022_7_11\GOPRO\GOPRO_Large_all\train_images"

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []

        for subfile in os.listdir(self.image_root):
            event_root = os.path.join(self.event_root, subfile)
            image_root = os.path.join(self.image_root, subfile, "imgs")
        img_list = os.listdir(image_root)
        eve_list = os.listdir(event_root)
        eve_list.sort()
        eve_list.sort()
        pic_num = len(img_list)

        for i in range(16, pic_num, 16):
            img1 = img_list[i - 16]
            img2 = img_list[i]
            img1path = os.path.join(image_root, img1)
            img2path = os.path.join(image_root, img2)
            events1 = eve_list[i - 16]
            events2 = eve_list[i - 15]
            events3 = eve_list[i - 14]
            events4 = eve_list[i - 13]
            events5 = eve_list[i - 12]
            events6 = eve_list[i - 11]
            events7 = eve_list[i - 10]
            events8 = eve_list[i - 9]
            events9 = eve_list[i - 8]
            events10 = eve_list[i - 7]
            events11 = eve_list[i - 6]
            events12 = eve_list[i - 5]
            events13 = eve_list[i - 4]
            events14 = eve_list[i - 3]
            events15 = eve_list[i - 2]
            events16 = eve_list[i - 1]
            e_flie = [os.path.join(event_root, events1), os.path.join(event_root, events2),
                      os.path.join(event_root, events3), os.path.join(event_root, events4),
                      os.path.join(event_root, events5), os.path.join(event_root, events6),
                      os.path.join(event_root, events7), os.path.join(event_root, events8),
                      os.path.join(event_root, events9), os.path.join(event_root, events10),
                      os.path.join(event_root, events11), os.path.join(event_root, events12),
                      os.path.join(event_root, events13), os.path.join(event_root, events14),
                      os.path.join(event_root, events15), os.path.join(event_root, events16)]
            for index in range(15):
                k = 15 - index
                img_gt = img_list[i - k]
                img_gt_path = os.path.join(image_root, img_gt)

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

        timestamp = (flag + 1) / 16
        eventpoints = []
        for file in events_path:
            eventpoints.append(torch.Tensor(np.load(file)))
        eventpoints = torch.vstack(eventpoints)

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

        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)

        x = eventpoints[:, 0:1] / the_size
        y = eventpoints[:, 1:2] / the_size
        p = eventpoints[:, 3:]
        eventpoints = np.concatenate([alltime, x, y, p], 1)
        events = torch.Tensor(eventpoints)

        num = h * w // 64 * 64
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
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        return img1, img2, img_gt, events, neighbors, h, w, timestamp

    def __len__(self):
        return len(self.eventslist)