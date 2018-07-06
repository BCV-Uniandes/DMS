# -*- coding: utf-8 -*-

"""DMN PyTorch Model tests."""

# Standard Lib imports
import os
import os.path as osp

# PyTorch imports
import torch
from torchvision.transforms import Compose, ToTensor, Normalize

# Local imports
from dmn_pytorch.models import DMN, BaseDMN
from dmn_pytorch.referit_loader import ReferDataset
from dmn_pytorch.utils.transforms import ResizeAnnotation, ResizeImage

# PyTest imports
import pytest


@pytest.fixture
def loader_fixture():
    target_transform = Compose([
        ResizeAnnotation(512),
    ])
    input_transform = Compose([
        ToTensor(),
        ResizeImage(512),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    path = osp.join(os.environ['HOME'], 'referit_data')
    split_path = osp.join(os.environ['HOME'], 'data')
    loader = ReferDataset(path, split_path, 'unc', input_transform,
                          target_transform,
                          'testA', -1)
    return loader

@pytest.fixture
def dmn_fixture_lowres(loader_fixture):
    params = {
        "dict_size": len(loader_fixture.corpus),
        "emb_size": 1000,
        "hid_size": 1000,
        "vis_size": 2688,
        "num_filters": 10,
        "mixed_size": 1000,
        "hid_mixed_size": 1000,
        "lang_layers": 3,
        "mixed_layers": 3,
        "backend": 'dpn92',
        "mix_we": True,
        "lstm": False,
        "pretrained": True,
        "extra": True,
        "high_res": False
    }
    return params

def test_dmn_forward_lowres(dmn_fixture_lowres, loader_fixture):
    dmn = BaseDMN(**dmn_fixture_lowres)
    dmn = dmn.cuda()
    for i in range(0, 10):
        img, mask, phrase = loader_fixture[i]
        img = img.cuda().unsqueeze(0)
        mask = mask.cuda()
        phrase = phrase.cuda().unsqueeze(0)
        with torch.no_grad():
            out = dmn(img, phrase)
        assert out.size == (img.size(-2) // 32, img.size(-1) // 32)
    del dmn
