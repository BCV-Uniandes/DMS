# -*- coding: utf-8 -*-

"""
QSegNet evaluation routines.
"""

# Standard lib imports
import time
import argparse
import os.path as osp

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from models import QSegNet
from referit_loader import ReferDataset
from utils.transforms import ResizePad, ToNumpy

# Other imports
import numpy as np

parser = argparse.ArgumentParser(
    description='Query Segmentation Network evaluation routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='../referit_data',
                    help='path to ReferIt splits data folder')
parser.add_argument('--snapshot', default='weights/qsegnet_unc_snapshot.pth',
                    help='path to weight snapshot file')
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of workers used in dataloading')
parser.add_argument('--dataset', default='unc', type=str,
                    help='dataset used to train QSegNet')
parser.add_argument('--split', default='testA', type=str,
                    help='name of the dataset split used to train')

# Training procedure settings
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--batch-size', default=3, type=int,
                    help='Batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# Model settings
parser.add_argument('--size', default=320, type=int,
                    help='image size')
parser.add_argument('--time', default=20, type=int,
                    help='maximum time steps per batch')
parser.add_argument('--emb-size', default=200, type=int,
                    help='word embedding dimensions')
parser.add_argument('--backend', default='densenet', type=str,
                    help='default backend network to initialize PSPNet')
parser.add_argument('--psp-size', default=1024, type=int,
                    help='number of input channels to PSPNet')
parser.add_argument('--num-features', '--features', default=512, type=int,
                    help='number of PSPNet output channels')
parser.add_argument('--lstm-layers', default=2, type=int,
                    help='number of LSTM stacked layers')
parser.add_argument('--vilstm-layers', default=1, type=int,
                    help='number of ViLSTM stacked layers')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

image_size = (args.size, args.size)

input_transform = Compose([
    ResizePad(image_size),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    ToNumpy(),
    ResizePad(image_size),
    ToTensor()
])

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=args.time)

loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True)

net = QSegNet(image_size, args.emb_size, args.size // 8,
              num_vilstm_layers=args.vilstm_layers,
              num_lstm_layers=args.lstm_layers,
              psp_size=args.psp_size,
              backend=args.backend,
              out_features=args.num_features,
              dict_size=len(refer.corpus))

net = nn.DataParallel(net)

if osp.exists(args.snapshot):
    print('Loading state dict')
    net.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    net.cuda()


def evaluate():
    step_size = 0.001
    thresholds = np.arange(0, 1 + step_size, step_size)
    net.eval()
    total_intersection = np.zeros(len(thresholds))
    total_union = np.zeros(len(thresholds))
    start_time = time.time()
    for batch_idx, (imgs, masks, words) in enumerate(loader):
        imgs = Variable(imgs, volatile=True)
        masks = masks.squeeze().cpu().numpy()
        words = Variable(words, volatile=True)

        if args.cuda:
            imgs = imgs.cuda()
            words = words.cuda()

        out = net(imgs, words)
        out = F.sigmoid(out)
        out = out.squeeze().data.cpu().numpy()

        batch_intersection, batch_union = intersection_and_union(
            out=out, target=masks, thresholds=thresholds)

        # Update total intersection and union
        total_intersection += batch_intersection
        total_union += batch_union

        if batch_idx % args.log_interval == 0:
            batch_iou = batch_intersection / batch_union
            max_batch_iou = np.amax(batch_iou)
            which_thresh_batch = thresholds[np.argmax(batch_iou)]

            partial_iou = total_intersection / total_union
            max_partial_iou = np.amax(partial_iou)
            which_thresh_partial = thresholds[np.argmax(partial_iou)]

            mean_batch_iou = np.mean(batch_iou)
            mean_partial_iou = np.mean(partial_iou)

            elapsed_time = time.time() - start_time
            print('({:5d}/{:5d}) | ms/batch {:.6f} |'
                  ' max batch IoU {:.6f} - threshold {:.3f} |'
                  ' max partial IoU {:.6f} - threshold {:.3f} |'
                  ' batch mIoU {:.6f} | partial mIoU {:.6f}'.format(
                      batch_idx, len(loader), elapsed_time * 1000,
                      max_batch_iou, which_thresh_batch,
                      max_partial_iou, which_thresh_partial,
                      mean_batch_iou, mean_partial_iou))
            start_time = time.time()


if __name__ == '__main__':
    print('Evaluating')
    evaluate()
