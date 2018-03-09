# -*- coding: utf-8 -*-

"""
QSegNet train routines. (WIP)
"""

# Standard lib imports
import os
import time
import argparse
import os.path as osp
from urllib.parse import urlparse

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from utils import AverageMeter
from utils.losses import IoULoss
from models import LangVisUpsample
from referit_loader import ReferDataset
from utils.misc_utils import VisdomWrapper
from utils.transforms import ResizeImage, ResizeAnnotation


parser = argparse.ArgumentParser(
    description='Query Segmentation Network training routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='../referit_data',
                    help='path to ReferIt splits data folder')
parser.add_argument('--split-root', type=str, default='data',
                    help='path to dataloader splits data folder')
parser.add_argument('--save-folder', default='weights/',
                    help='location to save checkpoint models')
parser.add_argument('--snapshot', default='weights/qseg_weights.pth',
                    help='path to weight snapshot file')
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of workers used in dataloading')
parser.add_argument('--dataset', default='unc', type=str,
                    help='dataset used to train QSegNet')
parser.add_argument('--split', default='train', type=str,
                    help='name of the dataset split used to train')
parser.add_argument('--val', default=None, type=str,
                    help='name of the dataset split used to validate')

# Training procedure settings
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--backup-iters', type=int, default=10000,
                    help='iteration interval to perform state backups')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--patience', default=2, type=int,
                    help='patience epochs for LR decreasing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--iou-loss', action='store_true',
                    help='use IoULoss instead of BCE')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='epoch number to resume')
parser.add_argument('--optim-snapshot', type=str,
                    default='weights/qsegnet_optim.pth',
                    help='path to optimizer state snapshot')
parser.add_argument('--old-weights', action='store_true', default=False,
                    help='load LangVisNet weights on a LangVisUpsample module')
parser.add_argument('--norm', action='store_true',
                    help='enable language/visual features L2 normalization')
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
parser.add_argument('--num-filters', default=1, type=int,
                    help='number of filters to learn')
parser.add_argument('--mixed-size', default=1000, type=int,
                    help='number of combined lang/visual features filters')
parser.add_argument('--hid-mixed-size', default=1005, type=int,
                    help='multimodal model hidden size')
parser.add_argument('--lang-layers', default=2, type=int,
                    help='number of SRU/LSTM stacked layers')
parser.add_argument('--mixed-layers', default=3, type=int,
                    help='number of mLSTM/mSRU stacked layers')
parser.add_argument('--backend', default='dpn92', type=str,
                    help='default backend network to LangVisNet')
parser.add_argument('--mix-we', action='store_true', default=False,
                    help='train linear layer filters based also on WE')
parser.add_argument('--lstm', action='store_true', default=False,
                    help='use LSTM units for RNN modules. Default SRU')
parser.add_argument('--high-res', action='store_true',
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
parser.add_argument('--langvis-freeze', action='store_true', default=False,
                    help='freeze low res model and train only '
                         'upsampling layers')
parser.add_argument('--no-embcube', action='store_true', default=False)

# Other settings
parser.add_argument('--visdom', type=str, default=None,
                    help='visdom URL endpoint')

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

# If we are in 'low res' mode, downsample the target
target_transform = Compose([
    # ToTensor(),
    ResizeAnnotation(args.size),
])

if args.high_res:
    target_transform = Compose([
        # ToTensor()
    ])

if args.batch_size == 1:
    args.time = -1

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=args.time)

train_loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True)

start_epoch = args.start_epoch

if args.val is not None:
    refer_val = ReferDataset(data_root=args.data,
                             dataset=args.dataset,
                             split=args.val,
                             transform=input_transform,
                             # annotation_transform=target_transform,
                             max_query_len=args.time)
    val_loader = DataLoader(refer_val, batch_size=args.batch_size)


if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)


net = LangVisUpsample(dict_size=len(refer.corpus),
                      emb_size=args.emb_size,
                      hid_size=args.hid_size,
                      vis_size=args.vis_size,
                      num_filters=args.num_filters,
                      mixed_size=args.mixed_size,
                      hid_mixed_size=args.hid_mixed_size,
                      lang_layers=args.lang_layers,
                      mixed_layers=args.mixed_layers,
                      backend=args.backend,
                      mix_we=args.mix_we,
                      lstm=args.lstm,
                      high_res=args.high_res,
                      upsampling_channels=args.upsamp_channels,
                      upsampling_mode=args.upsamp_mode,
                      upsampling_size=args.upsamp_size,
                      gpu_pair=args.gpu_pair,
                      upsampling_amplification=args.upsamp_amplification,
                      langvis_freeze=args.langvis_freeze,
                      no_embcube=args.no_embcube)

if osp.exists(args.snapshot):
    snapshot_dict = torch.load(args.snapshot)
    if args.old_weights:
        state = {}
        for weight_name in snapshot_dict.keys():
            state['langvis.' + weight_name] = snapshot_dict[weight_name]
        snapshot_dict = state
    net.load_state_dict(snapshot_dict)

if args.cuda:
    net.cuda()

if args.visdom is not None:
    visdom_url = urlparse(args.visdom)

    port = 80
    if visdom_url.port is not None:
        port = visdom_url.port

    print('Initializing Visdom frontend at: {0}:{1}'.format(
          args.visdom, port))
    vis = VisdomWrapper(server=visdom_url.geturl(), port=port)

    vis.init_line_plot('iteration_plt', xlabel='Iteration', ylabel='Loss',
                       title='Current QSegNet Training Loss',
                       legend=['Loss'])

    vis.init_line_plot('epoch_plt', xlabel='Epoch', ylabel='Loss',
                       title='Current QSegNet Epoch Loss',
                       legend=['Loss'])

    if args.val is not None:
        vis.init_line_plot('val_plt', xlabel='Iteration', ylabel='Loss',
                           title='Current QSegNet Validation Loss',
                           legend=['Loss'])

optimizer = optim.Adam(net.parameters(), lr=args.lr)

scheduler = ReduceLROnPlateau(
    optimizer, patience=args.patience)

if osp.exists(args.optim_snapshot):
    optimizer.load_state_dict(torch.load(args.optim_snapshot))
    # last_epoch = args.start_epoch

scheduler.step(args.start_epoch)

criterion = nn.BCEWithLogitsLoss()
if args.iou_loss:
    criterion = IoULoss()


def train(epoch):
    net.train()
    total_loss = AverageMeter()
    # total_loss = 0
    epoch_loss_stats = AverageMeter()
    # epoch_total_loss = 0
    start_time = time.time()
    for batch_idx, (imgs, masks, words) in enumerate(train_loader):
        imgs = Variable(imgs)
        masks = Variable(masks.squeeze())
        words = Variable(words)

        if args.cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
            words = words.cuda()

        if args.cuda and args.gpu_pair is not None:
            imgs = imgs.cuda(2*args.gpu_pair)
            masks = masks.cuda(2*args.gpu_pair)
            words = words.cuda(2*args.gpu_pair)

        optimizer.zero_grad()
        out_masks = net(imgs, words)
        out_masks = F.upsample(out_masks, size=(
            masks.size(-2), masks.size(-1)), mode='bilinear').squeeze()
        if args.gpu_pair is not None:
            masks = masks.cuda(2*args.gpu_pair + 1)
        loss = criterion(out_masks, masks)
        loss.backward()
        optimizer.step()

        total_loss.update(loss.data[0], imgs.size(0))
        epoch_loss_stats.update(loss.data[0], imgs.size(0))
        # total_loss += loss.data[0]
        # epoch_total_loss += total_loss

        if args.visdom is not None:
            cur_iter = batch_idx + (epoch - 1) * len(train_loader)
            vis.plot_line('iteration_plt',
                          X=torch.ones((1, 1)).cpu() * cur_iter,
                          Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                          update='append')

        if batch_idx % args.backup_iters == 0:
            filename = 'qsegnet_{0}_{1}_snapshot.pth'.format(
                args.dataset, args.split)
            filename = osp.join(args.save_folder, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = 'qsegnet_{0}_{1}_optim.pth'.format(
                args.dataset, args.split)
            optim_filename = osp.join(args.save_folder, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            # cur_loss = total_loss / args.log_interval
            print('[{:5d}] ({:5d}/{:5d}) | ms/batch {:.6f} |'
                  ' loss {:.6f} | lr {:.7f}'.format(
                      epoch, batch_idx, len(train_loader),
                      elapsed_time * 1000, total_loss.avg,
                      optimizer.param_groups[0]['lr']))
            total_loss.reset()

        # total_loss = 0
        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg

    if args.visdom is not None:
        vis.plot_line('epoch_plt',
                      X=torch.ones((1, 1)).cpu() * epoch,
                      Y=torch.Tensor([epoch_total_loss]).unsqueeze(0).cpu(),
                      update='append')
    return epoch_total_loss


def validate(epoch):
    net.eval()
    epoch_total_loss = AverageMeter()
    start_time = time.time()
    for batch_idx, (imgs, masks, words) in enumerate(val_loader):
        imgs = Variable(imgs, volatile=True)
        masks = Variable(masks.squeeze(), volatile=True)
        words = Variable(words, volatile=True)

        if args.cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
            words = words.cuda()

        out_masks = net(imgs, words)
        out_masks = F.upsample(out_masks, size=(
            masks.size(-2), masks.size(-1)), mode='bilinear').squeeze()
        loss = criterion(out_masks, masks)
        epoch_total_loss.update(loss.data[0], imgs.size(0))

    epoch_total_loss = epoch_total_loss.avg
    elapsed_time = time.time() - start_time
    print('[{:5d}] Validation | elapsed time (ms) {:.6f} |'
          ' loss {:.6f}'.format(
              epoch, elapsed_time * 1000, epoch_total_loss))
    if args.visdom is not None:
        vis.plot_line('val_plt',
                      X=torch.ones((1, 1)).cpu() * epoch,
                      Y=torch.Tensor([epoch_total_loss]).unsqueeze(0).cpu(),
                      update='append')

    return epoch_total_loss


if __name__ == '__main__':
    print('Beginning training')
    best_val_loss = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train(epoch)
            val_loss = train_loss
            if args.val is not None:
                val_loss = validate(epoch)
            scheduler.step(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| epoch loss {:.6f} |'.format(
                      epoch, time.time() - epoch_start_time, train_loss))
            print('-' * 89)
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                filename = osp.join(args.save_folder, 'qsegnet_weights.pth')
                torch.save(net.state_dict(), filename)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
