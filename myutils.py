import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Event
from turtle import right
from numpy import linspace
from pytorch_msssim import ssim_matlab as calc_ssim
import math
import os
import torch
import shutil

import matplotlib.pyplot as plt
import copy

from torchvision import transforms
import open3d as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

def init_meters():
    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims


def eval_psnr(output, gt, psnrs):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1) , val_range=1.)
        ssims.update(ssim)
        
        
def optical_flow_warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)


def save_checkpoint(state, directory, is_best,  filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth'))

def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar('Loss/%s/%s' % mode, loss, timestep)
    writer.add_scalar('PSNR/%s' % mode, psnr, timestep)
    writer.add_scalar('SSIM/%s' % mode, ssim, timestep)
    if mode == 'train':
        writer.add_scalar('lr', lr, timestep)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']







def display_neighbor(neighbor_idx,xyt,color=None):
    """
    neighbor_idx: the neighbor index
    xyt: the coordinates of the points
    plot_colors: custom color list
    """

    predictions=np.int64(np.zeros(xyt[:,-1].shape))
    for points_i in range(xyt.shape[0]):
        this_prediction=copy.deepcopy(predictions)
 
        this_prediction[ neighbor_idx[points_i]] = 1
        this_prediction[points_i]=2

        if color == None:
            Plot.draw_pc_sem_ins(xyt, this_prediction)
        else:
            Plot.draw_pc_sem_ins(xyt, this_prediction,color)


def To_norm(flow):
    if flow.max()-flow.min()!=0:
        image_optical_flow = (flow-flow.min())/(flow.max()-flow.min())*2-1
    else:
        image_optical_flow = flow
        
    return image_optical_flow




def events_split(evn_fea, events, t_stamp):
    b, c, n =evn_fea.shape
    left_fea  = []
    right_fea = []
    mask = []
    for bi in range(b):
        t = events[bi, :, 0]
        left_ind = t<t_stamp[bi]
        right_ind = ~left_ind
        left_fea.append(evn_fea[bi, :, left_ind])
        right_fea.append(evn_fea[bi,:, right_ind])
        mask.append(left_ind)
    mask = torch.stack(mask)


    return left_fea, right_fea, mask

def e2v(fea, events, e_ratio,img_ratio, mask, bins=24, c=32):
    b,_,h,w = img_ratio.shape
    s = max(h,w)
    layers=[]

    for bi in range(b):

        img_fea = torch.zeros(c, bins * h * w).to(events.device)
        img_num = torch.ones(1,  bins * h * w).to(events.device)


        maski = mask[bi]
        feai = fea[bi]
        xyt = events[bi, maski, :3]
        t_ratio  = e_ratio[bi,:, maski].reshape(-1)
        im_ratio = img_ratio[bi, ...].reshape(-1)

        if xyt.shape[0] == 0:
            layers.append(img_fea.reshape(c,bins,h,w))
            continue

        t_ind =(xyt[:,0] - xyt[:,0].min()) / (xyt[:,0].max() - xyt[:,0].min() + 1e-5)
        t_ind = torch.clamp( (t_ind * bins).long(), 0, bins-1)

        x_ind = (xyt[:, 1] * s ).long()
        y_ind = (xyt[:, 2] * s ).long()

        # ind = (bins * h * t_ind + h * y_ind + x_ind)

        ind = (h * w * t_ind + w * y_ind + x_ind)

        s_ratio = im_ratio[w * y_ind + x_ind]

        the_fea = t_ratio * s_ratio * feai

        img_fea.index_add_( 1, ind, the_fea.reshape(c,-1) )
        img_num.index_add_( 1, ind, t_ratio.reshape(1,-1) * 10 )

        final = 10*img_fea / img_num

        layers.append(final.reshape(c,bins,h,w))

    return torch.stack(layers).cuda()


def save_testing(img_out, img_gt,pic_i ) :
    # path=r'/media/root/LENOVO_USB_HDD/Result/HQF'
    path=r'/media/root/LENOVO_USB_HDD/Result/vimeo'
    b,c,h,w = img_out.shape
    imgs=[]
    for index in range(b):
        imgs = transforms.ToPILImage()(img_out[index])
        imgs_gt = transforms.ToPILImage()(img_gt[index])
        imgs.save(path+"/{:08d}".format(pic_i)+str(index)+"0.png")
        imgs_gt.save(path+"/{:08d}".format(pic_i)+str(index)+"1.png")







def to_mask(event_sequence, nb_of_time_bins=1,h=48,w=48,remapping_maps=None):


    voxel_grid = torch.zeros(h,w,dtype=torch.float32,device='cpu')

    if event_sequence.shape[0]==0:
        return voxel_grid

    voxel_grid_flat = voxel_grid.flatten()


    features =event_sequence
    x = features[:, 1]
    y = features[:, 2]
    polarity = features[:, -1].float()
    p2=torch.ones(polarity.shape)

    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            mask = (0 <= lim_x) & (0 <= lim_y)  & (lim_x <= w-1)   & (lim_y <= h-1)
            lin_idx = lim_x.long() + lim_y.long() * w
            voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=p2[mask].float())

    return voxel_grid



##------------------------------------3Dplot------------------------------------------------------------------------------

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb, window_name='Open3D'):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.visualization.draw_geometries([pc])
            return 0
        if pc_xyzrgb[:, 3:6].max() > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc], window_name=window_name)
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None, window_name='labels'):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)   #只有1
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))

        for id, semins in enumerate(sem_ins_labels):
            valid_ind=np.where(pc_sem_ins == semins)
      

            #找到有效ind
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = valid_xyz[:, 0].min()#np.min(valid_xyz[:, 0])
            xmax = valid_xyz[:, 0].max()#np.max(valid_xyz[:, 0])
            ymin = valid_xyz[:, 1].min()#np.min(valid_xyz[:, 1])
            ymax = valid_xyz[:, 1].max()#np.max(valid_xyz[:, 1])
            zmin = valid_xyz[:, 2].min()#np.min(valid_xyz[:, 2])
            zmax = valid_xyz[:, 2].max()#np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins, window_name=window_name)
        return Y_semins









##------------------------------------SizeAdapter------------------------------------------------------------------------------



def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)

class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0, self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self._pixels_pad_to_width:]
