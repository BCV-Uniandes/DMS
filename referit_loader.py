# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import cv2
import json
import uuid
import torch
import progressbar
import numpy as np
import os.path as osp
import scipy.io as sio
from referit import REFER
import torch.utils.data as data
from referit.refer import mask as cocomask

from utils import Corpus

cv2.setNumThreads(0)

class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        }
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, annotation_transform=None,
                 split='train', max_query_len=20):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.corpus = Corpus()
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.split = split

        self.dataset_root = osp.join(self.data_root, 'referit')
        self.im_dir = osp.join(self.dataset_root, 'images')
        self.mask_dir = osp.join(self.dataset_root, 'mask')
        self.split_dir = osp.join(self.dataset_root, 'splits')

        if self.dataset != 'referit':
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.mask_dir = osp.join(self.dataset_root, self.dataset, 'mask')

        if not self.exists_dataset():
            self.process_dataset()

        dataset_path = osp.join(self.split_root, self.dataset)
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        self.corpus = torch.load(corpus_path)

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

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

        splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        for split in splits:
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

        bar = progressbar.ProgressBar()
        for name in bar(im_list):
            im_filename = name.split('_', 1)[0] + '.jpg'
            if im_filename in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue
            if osp.exists(osp.join(self.im_dir, im_filename)):
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

    def process_coco(self, setname, dataset_folder):
        split_dataset = []
        vocab_file = osp.join(self.split_dir, 'vocabulary_Gref.txt')

        refer = REFER(
            self.dataset_root, **(
                self.SUPPORTED_DATASETS[self.dataset]['params']))

        refs = [refer.refs[ref_id] for ref_id in refer.refs
                if refer.refs[ref_id]['split'] == setname]

        refs = sorted(refs, key=lambda x: x['file_name'])

        if len(self.corpus) == 0:
            print('Saving dataset corpus dictionary...')
            corpus_file = osp.join(self.split_root, self.dataset, 'corpus.pth')
            self.corpus.load_file(vocab_file)
            torch.save(self.corpus, corpus_file)

        if not osp.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        bar = progressbar.ProgressBar()
        for ref in bar(refs):
            img_filename = 'COCO_train2014_{0}.jpg'.format(
                str(ref['image_id']).zfill(12))
            if osp.exists(osp.join(self.im_dir, img_filename)):
                h, w, _ = cv2.imread(osp.join(self.im_dir, img_filename)).shape
                seg = refer.anns[ref['ann_id']]['segmentation']
                rle = cocomask.frPyObjects(seg, h, w)
                mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
                mask = torch.from_numpy(mask)
                mask_file = str(uuid.uuid4()) + '.pth'
                mask_filename = osp.join(self.mask_dir, mask_file)
                if not osp.exists(mask_filename):
                    torch.save(mask, mask_filename)
                for sentence in ref['sentences']:
                    split_dataset.append((
                        img_filename, mask_file, sentence['sent']))

        output_file = '{0}_{1}.pth'.format(self.dataset, setname)
        torch.save(split_dataset, osp.join(dataset_folder, output_file))

    def pull_item(self, idx):
        img_file, mask_file, phrase = self.images[idx]

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)

        mask_path = osp.join(self.mask_dir, mask_file)
        mask = torch.load(mask_path)

        return img, mask, phrase

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask, phrase = self.pull_item(idx)
        if self.transform is not None:
            img = self.transform(img)
        if self.annotation_transform is not None:
            # mask = mask.byte() * 255
            mask = self.annotation_transform(mask)
        phrase = self.tokenize_phrase(phrase)
        return img, mask, phrase
