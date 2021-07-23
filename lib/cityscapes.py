#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

import sys

sys.path.append('./')
from lib.transforms import *


class LaneData(Dataset):
    def __init__(self, cfg, mode='train', *args, **kwargs):
        super(LaneData, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.cfg = cfg

        self.list = self._get_dataset_list(cfg.data_pth, mode)

        with open('./dataset/lane.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ])
        self.trans = Compose([
            Resize(cfg.image_resize),
            ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation),
            HorizontalFlip(),
            RandomScale(cfg.scales),
            RandomCrop(cfg.crop_size),
            RandomRotate(cfg.image_rotation)
        ])
        self.trans_val = transforms.Compose([
            transforms.Resize(cfg.image_resize)
        ])

    def __getitem__(self, idx):
        impth, lbpth = self.list[idx]
        img = cv2.imread(impth)[:, :, ::-1]
        img = Image.fromarray(img).convert('RGB')
        label = Image.fromarray(cv2.imread(lbpth, -1)).convert('L')
        # img = Image.open(impth).convert('RGB')
        # label = Image.open(lbpth).convert('L')
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        else:
            img = self.trans_val(img)
        imgs = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return imgs, label
        # return impth, imgs, label

    def __len__(self):
        return len(self.list)

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def _get_dataset_list(self, root, mode):
        img_list = [f.strip() for f in open(root + os.sep + mode + '.txt').readlines()]
        return [(root + os.sep + img_name.split(' ')[0], root + os.sep + img_name.split(' ')[1]) for img_name in
                img_list]


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from lib import config_factory

    cfg = config_factory['cityscapes']
    ds = LaneData(cfg, mode='train')
    dl = DataLoader(ds,
                    batch_size=4,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True)
    for impth, imgs, label in dl:
        # print(len(imgs))
        # for el in imgs:
        #    print(el.size())
        pass
