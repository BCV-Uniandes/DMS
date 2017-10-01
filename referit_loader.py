# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset
Define and group batches of images, segmentations and queries.

Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask

from word_utils import Corpus


class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {},
        'unc': {'dataset': 'refcoco', 'split_by': 'unc'},
        'unc+': {'dataset': 'refcoco+', 'split_by': 'unc'},
        'gref': {'dataset': 'refcocog', 'split_by': 'google'}
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, annotation_transform=None,
                 train=True, val=False, max_query_len=20):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.corpus = Corpus(max_len=max_query_len)
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.train = train
        self.val = val
        self.train_val = self.train and self.val
        self.test = not (self.train or self.val)

        self.im_dir = osp.join(self.data_root, 'images')
        self.mask_dir = osp.join(self.data_root, 'mask')
        self.split_dir = osp.join(self.data_root, 'splits')

        if not self.exists_dataset():
            self.process_dataset()

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def process_dataset(self):
        if self.dataset not in self.SUPPORTED_DATASETS:
            raise DatasetNotFoundError(
                'Dataset {0} is not supported by this loader'.format(
                    self.dataset))

        dataset_folder = osp.join(self.split_root, self.dataset)
        if not osp.exists(dataset_folder):
            os.makedirs(dataset_folder)

        if self.dataset == 'referit':
            data_func = self.process_referit
        else:
            data_func = self.process_coco

        for split in ('train', 'trainval', 'val', 'test'):
            print('Processing {0}: {1} set'.format(self.dataset, split))
            data_func(split, dataset_folder)

    def process_referit(self, setname, dataset_folder):
        split_dataset = []

        query_file = osp.join(
            self.split_dir, 'referit',
            'referit_query_{0}.json'.format(setname))
        vocab_file = osp.join(self.split_dir, 'vocabulary_referit.txt')

        query_dict = json.load(open(query_file))
        im_list = query_dict.keys()

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        for name in im_list:
            im_filename = name.split('_', 1)[0] + '.jpg'
            mask_mat_filename = osp.join(self.mask_dir, name + '.mat')
            mask_pth_filename = osp.join(self.mask_dir, name + '.pth')
            if osp.exists(mask_mat_filename):
                mask = sio.loadmat(mask_mat_filename)['segimg_t'] == 0
                mask = mask.astype(np.float64)
                mask = torch.from_numpy(mask)
                torch.save(mask, mask_pth_filename)
                os.remove(mask_mat_filename)
            for query in query_dict[name]:
                split_dataset.append((im_filename, name + '.pth', query))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))
