import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

from model.network import deeplabv3plus_resnet50 as S2M
from model.aggregate import aggregate_wbg_channel as aggregate
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by


class InteractiveManager:
    def __init__(self, model, image, mask, p_srb, n_srb):
        self.model = model

        self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0).cuda()
        self.mask = TF.to_tensor(mask).unsqueeze(0).cuda()

        h, w = self.image.shape[-2:]
        self.image, self.pad = pad_divide_by(self.image, 16)
        self.mask, _ = pad_divide_by(self.mask, 16)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = p_srb
        self.n_srb = n_srb

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.positive_mode = True
        self.need_update = True


    def run_s2m(self):
        # Convert scribbles to tensors
        Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rs = torch.cat([Rsp, Rsn], 1)
        Rs, _ = pad_divide_by(Rs, 16)

        # Use the network to do stuff
        inputs = torch.cat([self.image, self.mask, Rs], 1)
        _, mask = aggregate(torch.sigmoid(net(inputs)))

        # We don't overwrite current mask until commit
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy()[0,0] * 255).astype(np.uint8)

        if self.pad[2]+self.pad[3] > 0:
            np_mask = np_mask[self.pad[2]:-self.pad[3],:]
        if self.pad[0]+self.pad[1] > 0:
            np_mask = np_mask[:,self.pad[0]:-self.pad[1]]

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None



parser = ArgumentParser()
parser.add_argument('--image', default='ust_cat.jpg')
parser.add_argument('--model', default='saves/s2m.pth')
parser.add_argument('--mask', default=None)
parser.add_argument('--p_srb', default=None)
parser.add_argument('--n_srb', default=None)
parser.add_argument('--output', default='output_mask.jpg')
args = parser.parse_args()


def comp_image(image, mask, p_srb, n_srb):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 1
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    comp = (image*0.5 + color_mask*mask*0.5).astype(np.uint8)
    comp[p_srb>0.5, :] = np.array([0, 255, 0], dtype=np.uint8)
    comp[n_srb>0.5, :] = np.array([255, 0, 0], dtype=np.uint8)

    return comp


print('Usage: python interactive.py --image <image> --model <model> [Optional: --mask initial_mask]')


# network stuff
net = S2M()
net.load_state_dict(torch.load(args.model))
net = net.cuda().eval()
torch.set_grad_enabled(False)

# Reading stuff
image = cv2.imread(args.image, cv2.IMREAD_COLOR)
h, w = image.shape[:2]
if args.mask is None:
    mask = np.zeros((h, w), dtype=np.uint8)
else:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

if args.p_srb is None:
    p_srb = np.zeros((h, w), dtype=np.uint8)
else:
    p_srb = cv2.imread(args.p_srb, cv2.IMREAD_GRAYSCALE)
    p_srb[p_srb >= 250] = 0
    p_srb[p_srb > 0] = 1


if args.n_srb is None:
    n_srb = np.zeros((h, w), dtype=np.uint8)
else:
    n_srb = cv2.imread(args.n_srb, cv2.IMREAD_GRAYSCALE)
    n_srb[n_srb >= 250] = 0
    n_srb[n_srb > 0] = 1

manager = InteractiveManager(net, image, mask, p_srb, n_srb)


np_mask = manager.run_s2m()
display = comp_image(image, np_mask, manager.p_srb, manager.n_srb)
cv2.imwrite(args.output, display)
