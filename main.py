import collections
from pickle import TRUE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import myutils
from loss import Loss
from DataLoader.mydataloader import Goprosataset
from DataLoader.fullgoprotestloader import Goprotestdataset
from DataLoader.vimeoset import vimeotestset,vimeotrainset
from DataLoader.middleburyset import  middleburyset1,middleburyset3,middleburyset5
from DataLoader.HSBSRGBset import HSBSRGB_tripple,HSBSRGB_single
from DataLoader.HQFset import HQF1,HQF3
from DataLoader.HSRGB import HSset7,HSset5
from DataLoader.Gorpo import Gopro7,Gopro15
from torch.optim import Adam
from PIFnet import myNet
import argparse
from training import *
from testing import *
import collections


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="use cuda or not")
parser.add_argument("--load_checkpoints", type=bool, default=True, help="...")

parser.add_argument("--train_batchsize", type=int, default=2, help="the batchsize setting when training")
parser.add_argument("--test_batchsize", type=int, default=1, help="the batchsize setting when testing")

parser.add_argument("--epochs", type=int, default=30, help="the total epochs")
parser.add_argument("--init_epoch", type=int, default=0, help="the initial epoch")

parser.add_argument("--lr", type=int, default=1e-4, help="the initial learning rate")
parser.add_argument("--model", type=str, default="vimeo_model.pth", help="model path")
parser.add_argument("--datapath", type=str, default="", help="data path")
parser.add_argument("--workers", type=int, default=4, help=" ")

args = parser.parse_args()



if __name__ == "__main__":

    init_epoch = args.init_epoch
  
    data_path = args.datapath
    
    # traindataset = vimeodata_pathtrainset(data_path)
    # trainloader = DataLoader(traindataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    

 

    testdataset = vimeotestset(data_path)
    testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)



#--------------------------------------------------------------------------------------------------------#
    # testdataset = middleburyset1(data_path)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers,
    #                         pin_memory=True)
    # testdataset = middleburyset3(data_path)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)



 #--------------------------------------------------------------------------------------------------------#



# --------------------------------------------------------------------------------------------------------#
    # testdataset = HSBSRGB_tripple(data_path)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    # testdataset = HSBSRGB_single(data_path)
    # testloader = DataLoader(testdataset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)



    # traindataset = HSBSRGB_train(data_path)
    # trainloader = DataLoader(traindataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
#--------------------------------------------------------------------------------------------------------#



    criterion = Loss()

    device = torch.device('cuda:0' if args.cuda else 'cpu')

    mymodel= myNet().to(device)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        mymodel = torch.nn.DataParallel(mymodel).to(device)



    mymodel.train()
    optimizer = Adam(mymodel.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    best_psnr = 0

    if args.load_checkpoints:
        model_path =args.model
        checkpoint = torch.load(model_path, map_location = device)
        mymodel.load_state_dict(checkpoint)

      




    for epoch in range(init_epoch, args.epochs):
        # Vimeotrain(trainloader, epoch,  mymodel, criterion, optimizer,traindataset,args)
        test_loss, psnr, ssim = vimeotest(epoch, mymodel, criterion, testloader,device,args)

        # save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        myutils.save_checkpoint({
            'epoch': epoch,
            'state_dict': mymodel.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'lr': optimizer.param_groups[-1]['lr']
        }, './logs/train', is_best, 'ep%03d-psnr%.3f-ssim%.4f.pth' % (epoch + 1,psnr,ssim))
        lr_scheduler.step()

