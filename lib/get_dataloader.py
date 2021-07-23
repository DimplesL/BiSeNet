import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import os

from lib.sampler import RepeatedDistSampler
from lib.base_dataset import TransformationTrain, TransformationVal
from lib.cityscapes_cv2 import CityScapes
from lib.cityscapes import LaneData


def build_data_loader(cfg, mode='train', shuffle=False, distributed=True):
    ds = LaneData(cfg, mode=mode)
    drop_last = True if mode == 'train' else False
    shuffle = True if mode == 'train' else False
    batch_size = cfg.ims_per_gpu if mode == 'train' else cfg.eval_ims_per_gpu
    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not cfg.max_iter is None
            nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, cfg.nworkers])  # number of workers
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            # sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, cfg.eval_n_workers])  # number of workers
            sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=True,
        )
    else:
        nw = min([os.cpu_count(), cfg.batch_size if cfg.batch_size > 1 else 0, cfg.nworkers])
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=nw,
            pin_memory=True,
        )
    return dl


def get_data_loader(cfg, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(cfg.scales, cfg.cropsize)
        batchsize = cfg.ims_per_gpu
        annpath = cfg.train_im_anns
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not cfg.max_iter is None
            n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl
