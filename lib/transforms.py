# -*- coding: utf-8 -*-
"""
@author: Qiu Yurui
@contact: qiuyurui@maituan.com
@software: Pycharm
@file: transforms.py
@time: 2021/7/23 11:30 上午
@desc:
"""
# !/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        th, tw = self.size
        return dict(
            im=im.resize((tw, th), Image.BILINEAR),
            lb=lb.resize((tw, th), Image.NEAREST)
        )


class Resize_val(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        H, W = self.size
        w, h = im.size
        if w > W or h > H:
            scale = float(H) / h if w < 2 * h else float(W) / w
            w, h = int(round(scale * w)), int(round(scale * h))
        return dict(
            im=im.resize((w, h), Image.BILINEAR),
            lb=lb.resize((w, h), Image.NEAREST)
        )


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
            im=im.crop(crop),
            lb=lb.crop(crop)
        )


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        rotate_degree = random.random() * 2 * self.degree - self.degree

        return dict(
            im=im.rotate(rotate_degree, Image.BILINEAR),
            lb=lb.rotate(rotate_degree, Image.NEAREST)
        )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im=im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb=lb.transpose(Image.FLIP_LEFT_RIGHT),
                        )


class RandomScale(object):
    def __init__(self, scales=(1,), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im=im.resize((w, h), Image.BILINEAR),
                    lb=lb.resize((w, h), Image.NEAREST),
                    )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness > 0:
            self.brightness = [max(1 - brightness, 0), 1 + brightness]
        if not contrast is None and contrast > 0:
            self.contrast = [max(1 - contrast, 0), 1 + contrast]
        if not saturation is None and saturation > 0:
            self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im,
                    lb=lb,
                    )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W * ratio), int(H * ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb


if __name__ == '__main__':
    flip = HorizontalFlip(p=1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
