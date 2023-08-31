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



class middleburyset1(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'events')
        self.image_root = os.path.join(self.data_root, 'images')


        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        for sub in os.listdir(self.image_root):
            # if not sub == "Urban2":
            #     continue
            img_path=os.path.join(self.image_root,sub,'imgs')
            evn_path=os.path.join(self.event_root,sub)
            imgs_list=os.listdir(img_path)
            events_list= os.listdir(evn_path)
            img_len= len(imgs_list)
            for i in  range(2,img_len,2):
                img1 = imgs_list[i-2]
                img_gt=imgs_list[i-1]
                img2 = imgs_list[i]
                events1= events_list[i-2]
                events2= events_list[i-1]

                img1path = os.path.join(img_path, img1) 
                img2path = os.path.join(img_path,img2) 
                img_gt_path = os.path.join(img_path, img_gt) 
                e_flie = [ os.path.join(evn_path, events1) ,os.path.join(evn_path, events2) ]
                self.eventslist.append( e_flie )  # here
                self.image1list.append( img1path )  # here
                self.image2list.append( img2path )  # here
                self.imagegtlist.append( img_gt_path )  # here

        self.filelength = len(self.image1list)    #00001  *100


    def __getitem__(self, index):

        events_list = self.eventslist[index]

        img1path = self.image1list[index]
        img2path = self.image2list[index]
        img_gt_path = self.imagegtlist[index]
        # framei = self.framei[index] 

        eventpoints = []
        for file in events_list:
            eventpoints.append( torch.Tensor(np.load(file,allow_pickle=True)  ))
        eventpoints = torch.vstack(eventpoints)


        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)
        Im1= cv.cvtColor(Im1,cv.COLOR_BGR2RGB)
        Im2= cv.cvtColor(Im2,cv.COLOR_BGR2RGB)
        Igt= cv.cvtColor(Igt,cv.COLOR_BGR2RGB)


        h, w, _ = Im1.shape
        the_size =max(h, w)
        img1=Im1
        img2=Im2

        img_gt =Igt
        # checkdisplay_withimg(eventpoints,img1)
        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 
            
        #================================align=================================================================
        if h==388 and w==584:
            x = (eventpoints[:, 0:1]+4) /  the_size
            y = (eventpoints[:, 1:2]+2) / the_size
        elif h==480 and w==640:
            x = (eventpoints[:, 0:1]) /  the_size
            y = (eventpoints[:, 1:2]) / the_size
        else:
            print("wrong")
        p = eventpoints[:, 3:] 

        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        eventpoints = torch.Tensor( eventpoints)

        #================================add noise=================================================================
        # n = int(eventpoints.shape[0] *0.9)
        # n_t = []
        # n_x = []
        # n_y = []
        # n_p = []
        # for i in range(n):
        #     n_t.append(random.random())
        #     n_x.append(random.random()*w/the_size)
        #     n_y.append(random.random()*h/the_size)
        #     n_p.append(2 * random.randint(0, 1) - 1)
        # noise = np.concatenate(
        #     [np.array(n_t).reshape(-1, 1), np.array(n_x).reshape(-1, 1), np.array(n_y).reshape(-1, 1),
        #      np.array(n_p).reshape(-1, 1)], 1)
        # noise = torch.Tensor(noise)
        # eventpoints = torch.vstack([eventpoints, noise])

        #================================加噪声=================================================================

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

        elif eventpoints.shape[0] > num * 4:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 4, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

            
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        Timestamp = 0.5

        return img1, img2, img_gt, events, neighbors, h, w, Timestamp

    def __len__(self):
        return len(self.eventslist)




class middleburyset3(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'events')
        self.image_root = os.path.join(self.data_root, 'images')


        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag=[]
        for sub in os.listdir(self.image_root):
            img_path=os.path.join(self.image_root,sub,'imgs')
            evn_path=os.path.join(self.event_root,sub)
            imgs_list=os.listdir(img_path)
            events_list= os.listdir(evn_path)
            img_len= len(imgs_list)
            for i in  range(4,img_len,4):
                img1 = imgs_list[i-4]
                img2 = imgs_list[i]
                events1= events_list[i-4]
                events2= events_list[i-3]
                events3= events_list[i-2]
                events4= events_list[i-1]
                e_flie = [ os.path.join(evn_path, events1) ,os.path.join(evn_path, events2) ,
                            os.path.join(evn_path, events3) ,os.path.join(evn_path, events4) ]

                img1path = os.path.join(img_path, img1) 
                img2path = os.path.join(img_path,img2) 

                for index in range(3):
                    k = 3-index 
                    img_gt=imgs_list[i-k]
                    img_gt_path = os.path.join(img_path, img_gt) 

                    self.eventslist.append( e_flie )  # here
                    self.image1list.append( img1path )  # here
                    self.image2list.append( img2path )  # here
                    self.imagegtlist.append( img_gt_path )  # here
                    self.flag.append(index)
        self.filelength = len(self.image1list)    #00001  *100


    def __getitem__(self, index):

        events_list = self.eventslist[index]

        img1path = self.image1list[index]
        img2path = self.image2list[index]
        img_gt_path = self.imagegtlist[index]
        flag = self.flag[index]
        # framei = self.framei[index] 

        eventpoints = []
        for file in events_list:
            eventpoints.append( torch.Tensor(np.load(file,allow_pickle=True)  ))
        eventpoints = torch.vstack(eventpoints)


        Im1 = cv.imread(img1path)
        Im2 = cv.imread(img2path)
        Igt = cv.imread(img_gt_path)
        Im1= cv.cvtColor(Im1,cv.COLOR_BGR2RGB)
        Im2= cv.cvtColor(Im2,cv.COLOR_BGR2RGB)
        Igt= cv.cvtColor(Igt,cv.COLOR_BGR2RGB)


        h, w, _ = Im1.shape
        the_size =max(h, w)
        img1=Im1
        img2=Im2

        img_gt =Igt
        # checkdisplay_withimg(eventpoints,img1)
        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5) 
        if h==388 and w==584:
            x = (eventpoints[:, 0:1]+4) /  the_size
            y = (eventpoints[:, 1:2]+2) / the_size
        elif h==480 and w==640:
            x = (eventpoints[:, 0:1]) /  the_size
            y = (eventpoints[:, 1:2]) / the_size
        else:
            print("wrong")
        p = eventpoints[:, 3:] 

        eventpoints = np.concatenate([ alltime, x, y, p], 1)
        eventpoints = torch.Tensor( eventpoints)

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

        elif eventpoints.shape[0] > num * 4:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 4, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

            
        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        Timestamp =(flag+1)/4

        return img1, img2, img_gt, events, neighbors, h, w, Timestamp

    def __len__(self):
        return len(self.eventslist)


class middleburyset5(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.event_root = os.path.join(self.data_root, 'events')
        self.image_root = os.path.join(self.data_root, 'images')

        self.image1list = []
        self.image2list = []
        self.imagegtlist = []
        self.eventslist = []
        self.flag = []
        for sub in os.listdir(self.image_root):
            if not sub=="Beanbags":
                continue
            img_path = os.path.join(self.image_root, sub, 'imgs')
            evn_path = os.path.join(self.event_root, sub)
            imgs_list = os.listdir(img_path)
            events_list = os.listdir(evn_path)
            img_len = len(imgs_list)
            for i in range(6, img_len, 6):
                img1 = imgs_list[i - 6]
                img2 = imgs_list[i]
                events1 = events_list[i - 6]
                events2 = events_list[i - 5]
                events3 = events_list[i - 4]
                events4 = events_list[i - 3]
                events5 = events_list[i - 2]
                events6 = events_list[i - 1]
                e_flie = [os.path.join(evn_path, events1), os.path.join(evn_path, events2),
                          os.path.join(evn_path, events3), os.path.join(evn_path, events4),
                          os.path.join(evn_path, events5), os.path.join(evn_path, events6)]
                img1path = os.path.join(img_path, img1)
                img2path = os.path.join(img_path, img2)

                for index in range(5):
                    k = 5 - index
                    img_gt = imgs_list[i - k]
                    img_gt_path = os.path.join(img_path, img_gt)

                    self.eventslist.append(e_flie)  # here
                    self.image1list.append(img1path)  # here
                    self.image2list.append(img2path)  # here
                    self.imagegtlist.append(img_gt_path)  # here
                    self.flag.append(index)
        self.filelength = len(self.image1list)  # 00001  *100

    def __getitem__(self, index):

        events_list = self.eventslist[index]

        img1path = self.image1list[index]
        img2path = self.image2list[index]
        img_gt_path = self.imagegtlist[index]
        flag = self.flag[index]
        # framei = self.framei[index]

        eventpoints = []
        for file in events_list:
            eventpoints.append(torch.Tensor(np.load(file, allow_pickle=True)))
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


        # checkdisplay_withimg(eventpoints,img1)
        alltime = eventpoints[:, 2:3]
        if alltime.shape[0] > 1:
            # timestamp normalization
            alltime = (alltime - alltime.min()) / (alltime.max() - alltime.min() + 1e-5)
        if h == 388 and w == 584:
            x = (eventpoints[:, 0:1] + 4)
            y = (eventpoints[:, 1:2] + 2)
        elif h == 480 and w == 640:
            x = (eventpoints[:, 0:1])
            y = (eventpoints[:, 1:2])
        else:
            print("wrong")
        p = eventpoints[:, 3:]

        eventpoints = np.concatenate([alltime, x, y, p], 1)
        eventpoints = torch.Tensor(eventpoints)

        mask_vol = to_mask(eventpoints, h=h, w=w)
        e_mask = ~(mask_vol == 0)

        eventpoints[:, 1:2] = eventpoints[:, 1:2] / the_size
        eventpoints[:, 2:3] = eventpoints[:, 2:3] / the_size
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

        elif eventpoints.shape[0] > num * 4:
            n = np.linspace(0, eventpoints.shape[0] - 1, num * 4, dtype=int)
            events = eventpoints[n, :]
            neighbors = findknn(events)
        else:
            n = np.linspace(0, eventpoints.shape[0] - 1, eventpoints.shape[0] // 64 * 64, dtype=int)
            events = eventpoints[n, :]

            neighbors = findknn(events)

        img1 = torch.Tensor(img1.copy()).float().permute(2, 0, 1) / 255.0
        img2 = torch.Tensor(img2.copy()).float().permute(2, 0, 1) / 255.0
        img_gt = torch.Tensor(img_gt.copy()).float().permute(2, 0, 1) / 255.0

        Timestamp = (flag + 1) / 6

        return img1, img2, img_gt, events, neighbors, h, w, Timestamp,e_mask.repeat(3,1,1)

    def __len__(self):
        return len(self.eventslist)



