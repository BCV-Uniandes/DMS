# -*- coding: utf-8 -*-

"""
QSegNet train routines. (WIP)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from referit_loader import ReferDataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Scale, CenterCrop, ToPILImage, ToTensor, Normalize)

from utils.transforms import ResizePad, ToNumpy
from models import PSPNet

input_transform = Compose([
    # ToPILImage(),
    # CenterCrop(256),
    # Scale(136),
    ResizePad((304, 304)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    ToNumpy(),
    ResizePad((304, 304)),
    ToTensor()
])

refer = ReferDataset(data_root='/mnt/referit_data',
                     dataset='referit',
                     transform=input_transform,
                     annotation_transform=target_transform,
                     max_query_len=304)

loader = DataLoader(refer, batch_size=10, shuffle=True)

imgs, masks, words = next(iter(loader))
x = Variable(imgs, requires_grad=False, volatile=True)
x = x.cuda()

net = PSPNet(n_classes=1, backend='densenet', psp_size=1024)
net.cuda()
out = net(x).squeeze()

emb = nn.Embedding(len(refer.corpus), 500)
wemb = emb(words)
