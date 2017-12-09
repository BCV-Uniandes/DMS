# -*- coding: utf-8 -*-

"""
QSegNet Visdom visualization routines.
"""

# Standard lib imports
import argparse
import os.path as osp
from urllib.parse import urlparse

# PyTorch imports
import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from models import LangVisNet
from referit_loader import ReferDataset
from utils.transforms import ResizeImage

# Other imports
from visdom import Visdom

parser = argparse.ArgumentParser(
    description='Query Segmentation Network visualization routine')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='../referit_data',
                    help='path to ReferIt splits data folder')
parser.add_argument('--snapshot', default='weights/qseg_weights.pth',
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
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

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

# Other settings
parser.add_argument('--visdom', type=str,
                    default='http://visdom.margffoy-tuay.com',
                    help='visdom URL endpoint')
parser.add_argument('--env', type=str, default='langvis-vis',
                    help='name of the enviroment used to display'
                         'results on Visdom')
parser.add_argument('--num-images', type=int, default=30,
                    help='number of images to display on visdom')
parser.add_argument('--heatmap', action='store_true', default=False,
                    help='use heatmap to display mask')
parser.add_argument('--no-eval', action='store_true',
                    help='disable PyTorch evaluation mode')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.heatmap:
    args.batch_size = 1
    args.time = -1

# image_size = (args.size, args.size)

input_transform = Compose([
    ToTensor(),
    ResizeImage(args.size),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    # ToTensor(),
    ResizeImage(args.size),
])

refer = ReferDataset(data_root=args.data,
                     dataset=args.dataset,
                     split=args.split,
                     transform=input_transform,
                     # annotation_transform=target_transform,
                     max_query_len=args.time)

loader = DataLoader(refer, batch_size=args.batch_size, shuffle=True)

net = LangVisNet(dict_size=len(refer.corpus),
                 emb_size=args.emb_size,
                 hid_size=args.hid_size,
                 vis_size=args.vis_size,
                 num_filters=args.num_filters,
                 mixed_size=args.mixed_size,
                 hid_mixed_size=args.hid_mixed_size,
                 lang_layers=args.lang_layers,
                 mixed_layers=args.mixed_layers,
                 backend=args.backend,
                 lstm=args.lstm,
                 high_res=args.high_res,
                 upsampling_channels=args.upsamp_channels,
                 upsampling_mode=args.upsamp_mode,
                 upsampling_size=args.upsamp_size)

# net = nn.DataParallel(net)

if osp.exists(args.snapshot):
    print('Loading state dict')
    net.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    net.cuda()

visdom_url = urlparse(args.visdom)

port = 80
if visdom_url.port is not None:
    port = visdom_url.port

print('Initializing Visdom frontend at: {0}:{1}'.format(
      args.visdom, port))
vis = Visdom(server=visdom_url.geturl(), port=port)


def visualization():
    if not args.no_eval:
        net.eval()
    for i in range(0, args.num_images):
        imgs, masks, words = next(iter(loader))
        masks = target_transform(masks)
        imgs[:, 0] *= 0.229
        imgs[:, 1] *= 0.224
        imgs[:, 2] *= 0.225
        imgs[:, 0] += 0.485
        imgs[:, 1] += 0.456
        imgs[:, 2] += 0.406
        text = []
        text.append('{0}: Queries'.format(i))
        for j in range(0, words.size(0)):
            word = list(words[j])
            query = ' '.join(refer.untokenize_word_vector(word))
            text.append('{0}.{1}: {2}'.format(i, j, query))
        text = '<br>'.join(text)
        vis.text(text, env=args.env)
        vis.images(imgs.numpy(), env=args.env)
        vis.images(masks.numpy(), env=args.env)
        imgs = Variable(imgs, volatile=True)
        # masks = masks.squeeze().cpu().numpy()
        words = Variable(words, volatile=True)
        if args.cuda:
            imgs = imgs.cuda()
            words = words.cuda()
        out = net(imgs, words)
        out = F.upsample(out, size=(
            masks.size(-2), masks.size(-1)), mode='bilinear').squeeze()
        out = F.sigmoid(out)
        out = out.data.cpu().numpy()
        if args.heatmap:
            vis.heatmap(out.squeeze(), env=args.env)
        else:
            vis.images(out, env=args.env)

    print('Done')


if __name__ == '__main__':
    visualization()
