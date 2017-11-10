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
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from models import QSegNet
from referit_loader import ReferDataset
from utils.transforms import ResizePad, CropResize

# Other imports
import numpy as np
import progressbar

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
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--batch-size', default=3, type=int,
                    help='Batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-eval', action='store_true',
                    help='disable PyTorch evaluation mode')


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

target_transform = CropResize()

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=args.time)

# loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True)

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


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def evaluate():
    # if not args.no_eval:
    #    net.eval()
    net.train()
    score_thresh = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    start_time = time.time()
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for i in bar(range(0, len(refer))):
        img, mask, phrase = refer.pull_item(i)
        words = refer.tokenize_phrase(phrase)
        h, w, _ = img.shape
        img = input_transform(img)
        imgs = Variable(img, volatile=True).unsqueeze(0)
        mask = mask.squeeze()
        words = Variable(words, volatile=True).unsqueeze(0)

        if args.cuda:
            imgs = imgs.cuda()
            words = words.cuda()
            mask = mask.byte().cuda()
        out = net(imgs, words)
        out = F.sigmoid(out)
        # out = out.squeeze().data.cpu().numpy()
        out = out.squeeze()
        # out = (out >= score_thresh).astype(np.uint8)
        out = target_transform(out, (h, w))
        out = (out > score_thresh)
        out = out.squeeze().data

        try:
            inter, union = compute_mask_IU(out, mask)
        except AssertionError as e:
            continue
        cum_I += inter
        cum_U += union
        this_iou = inter / union
        for n_eval_iou in range(len(eval_seg_iou_list)):
            eval_seg_iou = eval_seg_iou_list[n_eval_iou]
            seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
        seg_total += 1

        if i != 0 and i % args.log_interval == 0:
        	print('Partial IoU:',cum_I/cum_U)

    # Evaluation finished. Compute total IoU and threshold that maximizes
    for n_eval_iou in range(len(eval_seg_iou_list)):
        print('precision@{:s} = {:.5f}'.format(
            str(eval_seg_iou_list[n_eval_iou]),
            seg_correct[n_eval_iou] / seg_total))

    print('Evaluation done. Elapsed time: {:.3f} (s) |'
          ' max IoU {:.6f}'.format((time.time() - start_time), cum_I / cum_U))


if __name__ == '__main__':
    print('Evaluating')
    evaluate()
