import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from myutils import Plot
from tqdm import tqdm
import myutils
from loss import Loss
from DataLoader.mydataloader import Goprosataset
from DataLoader.vimeoset import vimeotestset
from torch.optim import Adam
from PIFnet import myNet

import argparse




def Vimeotrain(trainloader, epoch, mymodel, criterion, optimizer,traindataset,args):
    mymodel.train()
    criterion.train()
    losses =0
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    with tqdm(total=len(traindataset)//args.train_batchsize, desc=f'Epoch {epoch + 1}/{args.epochs}',postfix=dict,mininterval=0.3) as pbar:
        for i, (img1, img2, img_gt, events, neighbors,Timestamp) in enumerate(trainloader):
            optimizer.zero_grad()

            img1 = img1.to(device)        #([b, 2, 3, 180, 320])
            img2 = img2.to(device)        #([b, 2, 3, 180, 320])
            img_gt = img_gt.float().to(device)  #([b, 3, 180, 320])
            events = events.to(device)    #[1, 200000, 4])
            neighbors = [x.to(device) for x in neighbors]
            Timestamp = [Timestamp.float().to(device)]

            img_out,flow1,flow2 = mymodel(events, img1, img2, neighbors,Timestamp)
            img_out = img_out[0]

            loss_sr, loss_flow ,loss_p= criterion(img_out, img_gt, img1, img2, flow1, flow2)
            loss = loss_sr + 0.1 * loss_flow + 0.01 * loss_p
            # loss = loss_sr
            losses += loss.item()

            loss.backward()
            optimizer.step()
            # pbar.set_postfix(**{'loss_sr': loss_sr.item(), 'loss_flow': loss_flow.item(), 'lr':  myutils.get_lr(optimizer)})
            pbar.set_postfix(**{'loss_sr': losses/(i+1),  'lr':  myutils.get_lr(optimizer)})
            pbar.update(1)