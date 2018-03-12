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
from models import LangVisUpsample
from referit_loader import ReferDataset
from utils.transforms import ResizeImage, ResizeAnnotation

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
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no-eval', action='store_true',
                    help='disable PyTorch evaluation mode')
parser.add_argument('--old-weights', action='store_true', default=False,
                    help='load LangVisNet weights on a LangVisUpsample module')
parser.add_argument('--gpu-pair', type=int, default=None,
                    help='gpu pair to use: either 0 (GPU0 and GPU1) or 1 (GPU2 and GPU3)')

# Model settings
parser.add_argument('--size', default=512, type=int,
                    help='image size')
parser.add_argument('--time', default=-1, type=int,
                    help='maximum time steps per batch')
parser.add_argument('--emb-size', default=1000, type=int,
                    help='word embedding dimensions')
parser.add_argument('--hid-size', default=1000, type=int,
                    help='language model hidden size')
parser.add_argument('--vis-size', default=2688, type=int,
                    help='number of visual filters')
parser.add_argument('--num-filters', default=10, type=int,
                    help='number of filters to learn')
parser.add_argument('--mixed-size', default=1000, type=int,
                    help='number of combined lang/visual features filters')
parser.add_argument('--hid-mixed-size', default=1005, type=int,
                    help='multimodal model hidden size')
parser.add_argument('--lang-layers', default=3, type=int,
                    help='number of SRU/LSTM stacked layers')
parser.add_argument('--mixed-layers', default=3, type=int,
                    help='number of mLSTM/mSRU stacked layers')
parser.add_argument('--backend', default='dpn92', type=str,
                    help='default backend network to LangVisNet')
parser.add_argument('--mix-we', action='store_true', default=False,
                    help='train linear layer filters based also on WE')
parser.add_argument('--lstm', action='store_true', default=False,
                    help='use LSTM units for RNN modules. Default SRU')
parser.add_argument('--high-res', action='store_true', default=True,
                    help='high res version of the output through '
                         'upsampling + conv')
parser.add_argument('--upsamp-channels', default=50, type=int,
                    help='number of channels in the upsampling convolutions')
parser.add_argument('--upsamp-mode', default='bilinear', type=str,
                    help='upsampling interpolation mode')
parser.add_argument('--upsamp-size', default=3, type=int,
                    help='upsampling convolution kernel size')
parser.add_argument('--upsamp-amplification', default=32, type=int,
                    help='upsampling scale factor')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

image_size = (args.size, args.size)

input_transform = Compose([
    ToTensor(),
    ResizeImage(args.size),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    # ToTensor(),
    ResizeAnnotation(args.size),
])

if args.high_res:
    target_transform = Compose([
        # ToTensor()
    ])

# target_transform = CropResize()

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=args.time)

# loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True)

net = LangVisUpsample()

# net = nn.DataParallel(net)

if osp.exists(args.snapshot):
    print('Loading state dict')
    snapshot_dict = torch.load(args.snapshot)
    if args.old_weights:
        state = {}
        for weight_name in snapshot_dict.keys():
            state['langvis.' + weight_name] = snapshot_dict[weight_name]
        snapshot_dict = state
    net.load_state_dict(snapshot_dict)

if args.cuda:
    net.cuda()


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union


def evaluate():
    net.train()
    if not args.no_eval:
        net.eval()
    score_thresh = np.concatenate([# [0],
                                   # np.logspace(start=-16, stop=-2, num=10,
                                   #             endpoint=True),
                                   np.arange(start=0.00, stop=0.96,
                                             step=0.025)]).tolist()
    cum_I = torch.zeros(len(score_thresh))
    cum_U = torch.zeros(len(score_thresh))
    eval_seg_iou_list = [.5, .6, .7, .8, .9]

    seg_correct = torch.zeros(len(eval_seg_iou_list), len(score_thresh))
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
            mask = mask.float().cuda()
        out = net(imgs, words)
        out = F.sigmoid(out)
        out = F.upsample(out, size=(
            mask.size(-2), mask.size(-1)), mode='bilinear').squeeze()
        # out = out.squeeze().data.cpu().numpy()
        # out = out.squeeze()
        # out = (out >= score_thresh).astype(np.uint8)
        # out = target_transform(out, (h, w))

        inter = torch.zeros(len(score_thresh))
        union = torch.zeros(len(score_thresh))
        for idx, thresh in enumerate(score_thresh):
            thresholded_out = (out > thresh).float().data
            try:
                inter[idx], union[idx] = compute_mask_IU(thresholded_out, mask)
            except AssertionError as e:
                inter[idx] = 0
                union[idx] = mask.sum()
                # continue

        cum_I += inter
        cum_U += union
        this_iou = inter / union

        for idx, seg_iou in enumerate(eval_seg_iou_list):
            for jdx in range(len(score_thresh)):
                seg_correct[idx, jdx] += (this_iou[jdx] >= seg_iou)

        seg_total += 1

        if i != 0 and i % args.log_interval == 0:
            temp_cum_iou = cum_I / cum_U
            _, which = torch.max(temp_cum_iou,0)
            which = which.numpy()
            print(' ')
            print('Accumulated IoUs at different thresholds:')
            print('+' + '-' * 34 + '+')
            print('| {:15}| {:15} |'.format('Thresholds', 'mIoU'))
            print('+' + '-' * 34 + '+')
            for idx, thresh in enumerate(score_thresh):
                this_string = '| {:<15.3E}| {:<15.8f} | <--' if idx == which else '| {:<15.3E}| {:<15.8f} |'
                print(this_string.format(thresh, temp_cum_iou[idx]))
            print('+' + '-' * 34 + '+')

    # Evaluation finished. Compute total IoUs and threshold that maximizes
    for jdx, thresh in enumerate(score_thresh):
        print('-' * 32)
        print('precision@X for Threshold {:<15.3E}'.format(thresh))
        for idx, seg_iou in enumerate(eval_seg_iou_list):
            print('precision@{:s} = {:.5f}'.format(
                str(seg_iou), seg_correct[idx, jdx] / seg_total))

    # Print final accumulated IoUs
    final_ious = cum_I / cum_U
    print('-' * 32 + '\n' + '')
    print('FINAL accumulated IoUs at different thresholds:')
    print('{:15}| {:15} |'.format('Thresholds', 'mIoU'))
    print('-' * 32)
    for idx, thresh in enumerate(score_thresh):
        print('{:<15.3E}| {:<15.13f} |'.format(thresh, final_ious[idx]))
    print('-' * 32)

    max_iou, max_idx = torch.max(final_ious, 0)
    max_iou = float(max_iou.numpy())
    max_idx = int(max_idx.numpy())

    # Print maximum IoU
    print('Evaluation done. Elapsed time: {:.3f} (s) '.format(
        time.time() - start_time))
    print('Maximum IoU: {:<15.13f} - Threshold: {:<15.13f}'.format(
        max_iou, score_thresh[max_idx]))


if __name__ == '__main__':
    print('Evaluating')
    evaluate()
