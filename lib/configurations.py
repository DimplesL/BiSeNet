#!/usr/bin/python
# -*- encoding: utf-8 -*-


class Config(object):
    def __init__(self):
        # dataset
        self.data_pth = '/home/hadoop-mtcv/cephfs/data/sujinming/0_lane_detection/dataset/20201201_image_1022_lane_list'
        # self.val_datapth = '/home/hadoop-mtcv/cephfs/data/sujinming/0_lane_detection/dataset/20201201_image_1022_lane_list'
        # self.test_datapth = '/home/hadoop-mtcv/cephfs/data/sujinming/0_lane_detection/dataset/20201201_image_1022_lane_list'
        self.n_workers = 8
        self.ims_per_gpu = 16
        self.batch_size = 32
        self.eval_ims_per_gpu = 2
        self.eval_n_workers = 4

        # data augment
        self.ignore_label = 255
        self.image_resize = (512, 1024)  # H, W
        self.crop_size = (960, 480)  # W, H
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.image_rotation = 10

        # model
        self.n_classes = 16
        self.aspp_global_feature = False
        self.model_type = 'bisenetv1',
        self.num_aux_heads = 2,

        # optimizer
        self.warmup_steps = 1000
        self.warmup_start_lr = 5e-6
        self.lr_start = 1e-2
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr_power = 0.9
        self.max_iter = 100000

        self.use_fp16 = True,
        self.use_sync_bn = True,
        self.respth = './res',

        # training control
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        self.flip = True
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.ohem_thresh = 0.7

        self.port = 32258

        ## eval control
        # self.eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
        self.eval_scales = (1.0,)
        # self.eval_flip = True
        self.eval_flip = False
        self.val_list = 'val_performance.txt'

        # display & save
        self.msg_iter = 10

        # refine
        self.current_it = 0
        self.current_epoch = 0
        self.snapshot = ''
