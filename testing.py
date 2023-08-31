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






def vimeotest(epoch, mymodel, criterion, testloader, device, args):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters()
    mymodel.eval()
    criterion.eval()

    with torch.no_grad():
        for i, (img1, img2, img_gt, events, neighbors, h, w, Timestamp) in enumerate(tqdm(testloader)):
            img1 = img1.to(device)  # ([b, 2, 3, 180, 320])
            img2 = img2.to(device)  # ([b, 2, 3, 180, 320])
            img_gt = img_gt.float().to(device)  # ([b, 3, 180, 320])
            events = events.to(device)  # [1, 200000, 4])
            Timestamp = [Timestamp.float().to(device)]
            neighbors = [x.to(device) for x in neighbors]

            img_out, _, _ = mymodel(events, img1, img2, neighbors, Timestamp)
            img_out = img_out[0]
            img_out = img_out[:, :, :h, :w]
            # Evaluate metrics
            img_out = torch.clamp(img_out,0,1)
            myutils.eval_metrics(img_out, img_gt, psnrs, ssims)
            myutils.save_testing(img_out, img_gt ,i)
            # e_mask=~e_mask
            #
            # img_outm = img_out*torch.tensor(e_mask,dtype=int).cuda() +torch.tensor(~e_mask,dtype=int).cuda()
            # img_gtm = img_gt*torch.tensor(e_mask,dtype=int).cuda() +torch.tensor(~e_mask,dtype=int).cuda()
            # myutils.eval_psnr(img_out[e_mask].reshape(1,-1), img_gt[e_mask].reshape(1,-1), psnrs)
            # myutils.save_testing(img_outm, img_gtm,i )

    # Print progress
    print(" PSNR: %f, SSIM: %f\n" %
          (psnrs.avg, ssims.avg))

    return losses.avg, psnrs.avg, ssims.avg


def vimeotest_tripple(epoch, mymodel, criterion, testloader, device, args):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims = myutils.init_meters()
    mymodel.eval()
    criterion.eval()

    with torch.no_grad():
        for i, (img1, img2, img_gt1, img_gt2, img_gt3, events, neighbors, h, w, Timestamp) in enumerate(
                tqdm(testloader)):
            img1 = img1.to(device)  # ([b, 2, 3, 180, 320])
            img2 = img2.to(device)  # ([b, 2, 3, 180, 320])
            img_gt1 = img_gt1.float().to(device)  # ([b, 3, 180, 320])
            img_gt2 = img_gt2.float().to(device)  # ([b, 3, 180, 320])
            img_gt3 = img_gt3.float().to(device)  # ([b, 3, 180, 320])
            events = events.to(device)  # [1, 200000, 4])
            neighbors = [x.to(device) for x in neighbors]
            Timestamp = [t.float().to(device) for t in Timestamp]
            img_gt = [img_gt1, img_gt2, img_gt3]

            img_out, _, _ = mymodel(events, img1, img2, neighbors, Timestamp)
            for index, img in enumerate(img_out):
                img = img[:, :, :h, :w]
                img = torch.clamp(img, 0, 1)
                myutils.eval_metrics(img, img_gt[index], psnrs, ssims)
                myutils.save_testing(img, img_gt[index], i * 3 + index)
                # Print progress
    print(" PSNR: %f, SSIM: %f\n" %
          (psnrs.avg, ssims.avg))

    return losses.avg, psnrs.avg, ssims.avg

