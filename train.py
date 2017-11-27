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
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from models import LangVisNet
from utils import AverageMeter
from utils.losses import IoULoss
from utils.transforms import ResizePad
from utils.misc_utils import VisdomWrapper
from referit_loader import ReferDataset, collate_fn


parser = argparse.ArgumentParser(
    description='Query Segmentation Network training routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='../referit_data',
                    help='path to ReferIt splits data folder')
parser.add_argument('--save-folder', default='weights/',
                    help='location to save checkpoint models')
parser.add_argument('--snapshot', default='weights/qseg_weights.pth',
                    help='path to weight snapshot file')
parser.add_argument('--num-workers', default=2, type=int,
                    help='number of workers used in dataloading')
parser.add_argument('--dataset', default='referit', type=str,
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
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--milestones', default='10,20,30', type=str,
                    help='milestones (epochs) for LR decreasing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--iou-loss', action='store_true',
                    help='use IoULoss instead of BCE')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='epoch number to resume')
parser.add_argument('--optim-snapshot', type=str,
                    default='weights/qsegnet_optim.pth',
                    help='path to optimizer state snapshot')
parser.add_argument('--norm', action='store_true',
                    help='enable language/visual features L2 normalization')

# Model settings
parser.add_argument('--size', default=1024, type=int,
                    help='image size')
parser.add_argument('--time', default=-1, type=int,
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
parser.add_argument('--dropout', default=0.2, type=float,
                    help='dropout constant to LSTM output layer')

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
    ResizePad(image_size),
    # ResizeImage(args.size),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    # ToNumpy(),
    # ToTensor(),
    ResizePad(image_size),
])

if args.batch_size == 1:
    args.time = -1

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=args.time)

train_loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn)

start_epoch = args.start_epoch

if args.val is not None:
    refer_val = ReferDataset(data_root=args.data,
                             dataset=args.dataset,
                             split=args.val,
                             transform=input_transform,
                             # annotation_transform=target_transform,
                             max_query_len=args.time)
    val_loader = DataLoader(refer_val, batch_size=args.batch_size,
                            collate_fn=collate_fn)


if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

# net = QSegNet(image_size, args.emb_size, args.size // 8,
#               num_vilstm_layers=args.vilstm_layers,
#               num_lstm_layers=args.lstm_layers,
#               psp_size=args.psp_size,
#               backend=args.backend,
#               out_features=args.num_features,
#               dropout=args.dropout,
#               dict_size=len(refer.corpus),
#               norm=args.norm)

net = LangVisNet(dict_size=len(refer.corpus))

net = nn.DataParallel(net)

if osp.exists(args.snapshot):
    net.load_state_dict(torch.load(args.snapshot))

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

scheduler = MultiStepLR(
    optimizer, milestones=[int(x) for x in args.milestones.split(',')])

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
        masks = Variable(masks.float().squeeze())
        words = Variable(words)

        if args.cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
            words = words.cuda()

        print(imgs.size())
        print(masks.size())
        print(words.size())

        optimizer.zero_grad()
        out_masks = net(imgs, words)
        # out_masks = F.upsample(out_masks, size=(
        #     masks.size(-2), masks.size(-1)), mode='bilinear').squeeze()
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
                      scheduler.get_lr()[0]))
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
        masks = Variable(masks, volatile=True)
        words = [Variable(w, volatile=True) for w in words]

        if args.cuda:
            imgs = imgs.cuda()
            masks = masks.cuda()
            words = [w.cuda() for w in words]

        out_masks = net(imgs, words)
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
            scheduler.step()
            train_loss = train(epoch)
            val_loss = train_loss
            if args.val is not None:
                val_loss = validate(epoch)
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
