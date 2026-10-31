
"""
Common modules
"""
import torch.fft

import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.post_process import non_max_suppression,non_max_suppression_ft
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import time_sync
import torch.nn.functional as F

GLOBAL_MASK_DICT = {}


GLOBAL_POINT_DICT = {}


GLOBAL_POINT_FEATURE_DICT = {}


GLOBAL_POINT_MOTION_DICT = {}


GLOBAL_MACRO_DETECT_DICT = {}


GLOBAL_FORMATION_SUPPORT_DICT = {}


GLOBAL_FOURIER_PROPOSAL_DICT = {}


GLOBAL_CANDIDATE_GEOMETRY_DICT = {}


GLOBAL_ORACLE_PERIPHERAL_DICT = {}


GLOBAL_ORACLE_FOURIER_BASE_DICT = {}


def probe_grad(name):


    def hook(grad):
        if grad is None:
            return


        if torch.isnan(grad).any():
            print(f"[NAN EXPLOSION] Layer: {name} | Grad has NaN!")


        elif torch.isinf(grad).any():
            print(f"[INF EXPLOSION] Layer: {name} | Grad has Inf!")


        else:
            max_val = grad.abs().max().item()
            if max_val > 1000:
                print(f"[HIGH GRAD] Layer: {name} | Max Abs Grad: {max_val:.4f}")


    return hook


def autopad(k, p=None):

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn. Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DetachedConv(Conv):
    """A Conv block whose input is read in forward but isolated in backward."""

    def forward(self, x):


        source = x.detach() if getattr(self, 'detach_input', True) else x
        return super().forward(source)

    def forward_fuse(self, x):
        source = x.detach() if getattr(self, 'detach_input', True) else x
        return super().forward_fuse(source)


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)


        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CAM(nn.Module):
    def __init__(self, c1,c2):
        super(CAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)

    def forward(self, x):
        out = self.channel_attention(x) * x
        return out


class SAM(nn.Module):
    def __init__(self, c1,c2):
        super(SAM, self).__init__()
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.spatial_attention(x) * x
        return out

class CSPCAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = CAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)


class CSPSAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = SAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)


class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):

    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CSPC(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = CBAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)


class C3TR(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)


    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
                                  GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(b, c * s * s, h // s, w // s)


class Expand(nn.Module):

    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(b, c // s ** 2, h * s, w * s)


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class ConcatDetachSkip(nn.Module):
    """Concatenate a trainable branch with detached shared-backbone skip features."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        if getattr(self, 'detach_skip', True):
            x = [x[0], *(feature.detach() for feature in x[1:])]
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):

    def __init__(self, weights='weights/best.pt', device=None, dnn=True):


        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)
        pt, onnx, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)
        jit = pt and 'torchscript' in w.lower()
        stride, names = 64, [f'class{i}' for i in range(1000)]

        if jit:
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])
                stride, names = int(d['stride']), d['names']
        elif pt:
            from models.experimental import attempt_load
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())
            names = model.module.names if hasattr(model, 'module') else model.names
            try:
                mask_dir = model.module.mask_dir if hasattr(model, 'module') else model.mask_dir
            except Exception:
                mask_dir = [0] * len(names)
            cms_s = False
            pts = (model.module.get_module_byname('Detect2') if hasattr(model, 'module') else model.get_module_byname('Detect2')) is not None
            cms_num = 0
            for key, num in zip(['SubclassInfer', 'ValueInfer'], [1, 'nv']):
                p = model.module.get_module_byname(key) if hasattr(model, 'module') else model.get_module_byname(key)
                if p:
                    if isinstance(num, int):
                        cms_num += num
                        cms_s = True
                    else:
                        cms_num += getattr(p, num)
                del p
            p = model.module.get_module_byname('FTInfer') if hasattr(model, 'module') else model.get_module_byname('FTInfer')
            ft_infer = p is not None
            ft_length = p.ft_length if ft_infer else 0
            del p
        self.__dict__.update(locals())

    def forward(self, im, augment=False, visualize=False, val=False):

        b, ch, h, w = im.shape
        if self.pt:
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)


            if val:
                out, train_out = [], []

                for t in y:


                    if isinstance(t, dict):
                        out.append(t)
                    else:
                        out.append(t[0])
                        if len(t) > 1:
                            train_out.extend(t[1])
                return out, train_out

            return y

        y = torch.tensor(y)
        return (y, []) if val else y


    def get_module_byname(self, name:str):
        return self.model.module.get_module_byname(name) if hasattr(self.model, 'module') else self.model.get_module_byname(name)

class AutoShape(nn.Module):

    conf = 0.25
    iou = 0.45
    classes = None
    multi_label = False
    max_det = 1000

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')
        return self

    def _apply(self, fn):

        self = super()._apply(fn)
        m = self.model.get_module_byname('Detect')
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):


        t = [time_sync()]
        p = next(self.model.parameters())
        if isinstance(imgs, torch.Tensor):
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)


        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, (str, Path)):
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = (size / max(s))
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):

            y = self.model(x, augment, profile)[0]
            t.append(time_sync())


            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:

    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.files = files
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]
        self.n = len(self.pred)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))
        self.s = shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)
        self.display(save=True, save_dir=save_dir)

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)

    def render(self):
        self.display(render=True)
        return self.imgs

    def pandas(self):

        new = copy(self)
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):

        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


class SpectralChannelAttention(nn.Module):
    """ [Frequency] Content-Aware Spectral Gating """
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1 * 2, c1 * 2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 * 2 // reduction, c1 * 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_cat):

        b, c2, _, _ = x_cat.size()
        y = self.avg_pool(x_cat).view(b, c2)
        y = self.fc(y).view(b, c2, 1, 1)
        return x_cat * y.expand_as(x_cat)

import torch.fft

class AdvancedFrequencyBranch(nn.Module):
    """Stable frequency-domain enhancement branch."""
    def __init__(self, c1):
        super().__init__()
        hidden = max(c1 // 4, 1)
        self.freq_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = nn.InstanceNorm2d(c1, affine=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        orig_dtype = x.dtype
        x_fp32 = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        fft_feat = torch.fft.rfft2(x_fp32, norm='ortho')
        amplitude = torch.abs(fft_feat)
        amp_weight = self.freq_weight(amplitude.to(orig_dtype)).float()
        fft_feat_filtered = fft_feat * amp_weight

        spatial_recovered = torch.fft.irfft2(
            fft_feat_filtered,
            s=(x.size(-2), x.size(-1)),
            norm='ortho',
        )
        spatial_recovered = torch.nan_to_num(spatial_recovered, nan=0.0, posinf=1e4, neginf=-1e4)
        norm_dtype = self.norm.weight.dtype if self.norm.weight is not None else orig_dtype
        spatial_recovered = self.norm(spatial_recovered.to(norm_dtype)).to(orig_dtype)
        alpha = 0.1 * torch.tanh(self.alpha).to(orig_dtype)
        return x + alpha * spatial_recovered

class SpatialGatingBranch(nn.Module):



    def __init__(self, c1):
        super().__init__()

        self.global_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )


        self.local_reduce = nn.Conv2d(c1, c1 // 4, 1, bias=False)
        self.local_center = nn.Conv2d(c1 // 4, c1 // 4, kernel_size=3, padding=1, bias=False)
        self.local_surround = nn.Conv2d(c1 // 4, c1 // 4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.local_out = nn.Sequential(
            nn.Conv2d(c1 // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        global_mask = self.global_conv(torch.cat([avg_out, max_out], dim=1))


        loc_x = self.local_reduce(x)

        loc_diff = F.relu(self.local_center(loc_x) - self.local_surround(loc_x))
        local_mask = self.local_out(loc_diff)


        omni_mask = torch.max(global_mask, local_mask)

        return x * omni_mask

class TrajectoryRelativeGate(nn.Module):
    """Motion regularity gate based on relative trajectory consistency."""
    def __init__(self, c1, seq_len):
        super().__init__()
        self.seq_len = seq_len
        reduced = max(4, c1 // 4)
        steps = max(seq_len - 2, 1)
        self.reduce = nn.Conv2d(c1, reduced, 1)
        self.bg_extractor = nn.AvgPool2d(kernel_size=15, stride=1, padding=7)
        self.group_support = nn.Conv2d(steps, steps, 3, padding=1, groups=steps)
        self.project = nn.Sequential(
            nn.Conv2d(steps, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_seq_flat):
        b, ct, h, w = x_seq_flat.shape
        c = ct // self.seq_len
        x_5d = x_seq_flat.view(b, self.seq_len, c, h, w)
        x_reduced = self.reduce(x_5d.reshape(-1, c, h, w)).view(b, self.seq_len, -1, h, w)
        deltas = x_reduced[:, 1:] - x_reduced[:, :-1]
        if deltas.shape[1] < 2:
            return torch.ones(b, 1, h, w, device=x_seq_flat.device, dtype=x_seq_flat.dtype)

        B, T, C, H, W = deltas.shape
        deltas_flat = deltas.reshape(B * T, C, H, W)
        bg_motion = self.bg_extractor(deltas_flat).view(B, T, C, H, W)
        rel_deltas = deltas - bg_motion

        d1 = rel_deltas[:, :-1].float()
        d2 = rel_deltas[:, 1:].float()
        dot_prod = torch.sum(d1 * d2, dim=2)
        norm_d1 = torch.sqrt(torch.sum(d1 ** 2, dim=2) + 1e-4)
        norm_d2 = torch.sqrt(torch.sum(d2 ** 2, dim=2) + 1e-4)
        cos_sim = dot_prod / (norm_d1 * norm_d2 + 1e-6)
        regularity_score = torch.nan_to_num(cos_sim, nan=0.0, posinf=1.0, neginf=-1.0)
        regularity_score = (regularity_score + 0.5).clamp(0, 1)

        target_dtype = self.group_support.weight.dtype
        regularity_score = self.group_support(regularity_score.to(target_dtype))
        return self.project(regularity_score).to(x_seq_flat.dtype)

class MotionConsistencyBranch(nn.Module):
    def __init__(self, c1, seq_len=5):
        super().__init__()
        self.seq_len = seq_len
        self.motion_fuse = nn.Sequential(
            nn.Conv2d(c1 * seq_len, c1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        nn.init.constant_(self.motion_fuse[0].bias, -1.0)
        self.bg_extractor = nn.AvgPool2d(kernel_size=15, stride=1, padding=7)

    def compute_cos_sim(self, d1, d2):
        d1 = torch.nan_to_num(d1.float(), nan=0.0, posinf=10.0, neginf=-10.0)
        d2 = torch.nan_to_num(d2.float(), nan=0.0, posinf=10.0, neginf=-10.0)
        dot_prod = torch.sum(d1 * d2, dim=2, keepdim=True)
        norm_d1 = torch.sqrt(torch.sum(d1 ** 2, dim=2, keepdim=True) + 1e-6)
        norm_d2 = torch.sqrt(torch.sum(d2 ** 2, dim=2, keepdim=True) + 1e-6)
        cos_sim = dot_prod / (norm_d1 * norm_d2 + 1e-8)
        cos_sim = torch.nan_to_num(cos_sim, nan=0.0, posinf=1.0, neginf=-1.0)
        valid_motion_mask = (norm_d1 > 0.05) & (norm_d2 > 0.05)
        return cos_sim * valid_motion_mask.float()

    def forward(self, x_seq_flat):
        b, ct, h, w = x_seq_flat.shape
        c = ct // self.seq_len
        x_5d = x_seq_flat.view(b, self.seq_len, c, h, w)
        motion_feat = torch.nan_to_num(self.motion_fuse(x_seq_flat), nan=0.0, posinf=10.0, neginf=-10.0)

        step_deltas = x_5d[:, 1:, ...] - x_5d[:, :-1, ...]
        step_deltas = torch.where(step_deltas.abs() > 0.05, step_deltas, torch.zeros_like(step_deltas))

        trajectory_energy = torch.mean(torch.abs(step_deltas), dim=1, keepdim=True)
        base_motion_mask = torch.mean(trajectory_energy, dim=2, keepdim=True)
        base_motion_mask = torch.clamp(base_motion_mask * 10.0, 0.0, 1.0)

        B, Tm1, C, H, W = step_deltas.shape
        bg_motion = self.bg_extractor(step_deltas.reshape(B * Tm1, C, H, W)).view(B, Tm1, C, H, W)
        rel_deltas = torch.nan_to_num(step_deltas - bg_motion, nan=0.0, posinf=10.0, neginf=-10.0)

        if Tm1 >= 2:
            d1_s1 = rel_deltas[:, :-1]
            d2_s1 = rel_deltas[:, 1:]
            score_s1 = torch.mean(self.compute_cos_sim(d1_s1, d2_s1), dim=1, keepdim=True)
        else:
            score_s1 = torch.ones_like(base_motion_mask)

        if self.seq_len >= 5:
            d1_s2 = (x_5d[:, 2, ...] - x_5d[:, 0, ...]).unsqueeze(1)
            d2_s2 = (x_5d[:, 4, ...] - x_5d[:, 2, ...]).unsqueeze(1)
            bg_s2_a = torch.sum(bg_motion[:, 0:2, ...], dim=1, keepdim=True)
            bg_s2_b = torch.sum(bg_motion[:, 2:4, ...], dim=1, keepdim=True)
            score_s2 = self.compute_cos_sim(d1_s2 - bg_s2_a, d2_s2 - bg_s2_b)
        else:
            score_s2 = score_s1

        direction_score = torch.max(score_s1, score_s2)
        direction_score = torch.clamp(direction_score, 0.0, 1.0) ** 2
        direction_score = F.avg_pool2d(direction_score.squeeze(1), kernel_size=3, stride=1, padding=1).unsqueeze(1)

        final_motion_mask = torch.clamp(base_motion_mask * direction_score.to(base_motion_mask.dtype) * 4.0, 0.0, 1.0)
        mask_4d = final_motion_mask.view(b, 1, h, w)
        mask_4d = F.max_pool2d(mask_4d, kernel_size=3, stride=1, padding=1)
        final_motion_mask_5d = mask_4d.unsqueeze(1).repeat(1, self.seq_len, 1, 1, 1)
        return motion_feat, final_motion_mask_5d


from torchvision.ops import DeformConv2d, deform_conv2d


def clean_grad_hook(grad):
    if grad is None:
        return None
    if torch.isnan(grad).any() or torch.isinf(grad).any():

        return torch.zeros_like(grad)
    return torch.clamp(grad, -10.0, 10.0)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os


class PointFeatureGate(nn.Module):
    def __init__(self, c1, max_gain=0.15, detach_prior=False, power=1.0,
                 gain_init=-2.0):
        super().__init__()
        self.c1 = c1
        self.max_gain = float(max_gain)
        self.detach_prior = bool(detach_prior)
        self.power = max(float(power), 1.0)
        hidden = max(c1 // 8, 8)
        self.point_embed = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.feature_delta = nn.Sequential(
            nn.Conv2d(c1 + hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.gain = nn.Parameter(torch.full((1,), float(gain_init)))

    def forward(self, x):
        point_logits = GLOBAL_POINT_DICT.get(x.device, None)
        if point_logits is None:
            return x
        point_prior = torch.sigmoid(point_logits).pow(self.power)
        if self.detach_prior:
            point_prior = point_prior.detach()
        point_prior = F.interpolate(point_prior, size=x.shape[-2:], mode='bilinear', align_corners=False)
        point_feature = self.point_embed(point_prior.to(dtype=x.dtype))
        delta = self.feature_delta(torch.cat((x, point_feature), dim=1))
        gain = self.max_gain * torch.sigmoid(self.gain)
        return x + gain * delta


class PeripheralPointFourierFusion(nn.Module):
    """Inject a detached peripheral-point envelope into a Fourier feature level.

    The prior deliberately emphasizes confident points that are both far from
    the response centroid and locally isolated.  A differentiable directional
    envelope then converts those sparse responses into a cluster-region prior.
    Only the Fourier feature passed to this module is changed.
    """

    def __init__(self, c1, max_gain=0.15, detach_prior=True, power=2.0,
                 gain_init=-2.0, radial_gain=1.0, isolation_gain=1.0,
                 directions=32, point_threshold=0.10, temperature=0.05,
                 radius_padding=0.5):
        super().__init__()
        self.c1 = c1
        self.max_gain = float(max_gain)
        self.detach_prior = bool(detach_prior)
        self.power = max(float(power), 1.0)
        self.radial_gain = float(radial_gain)
        self.isolation_gain = float(isolation_gain)
        self.directions = max(int(directions), 8)
        self.point_threshold = float(point_threshold)
        self.temperature = max(float(temperature), 1e-3)
        self.radius_padding = float(radius_padding)

        hidden = max(c1 // 8, 8)
        self.point_embed = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.feature_delta = nn.Sequential(
            nn.Conv2d(c1 + hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.gain = nn.Parameter(torch.full((1,), float(gain_init)))

    def _resize_probability(self, point_logits, size):
        probability = torch.sigmoid(point_logits.float()).pow(self.power)
        if self.detach_prior:
            probability = probability.detach()
        if probability.shape[-2:] == size:
            return probability
        if probability.shape[-2] >= size[0] and probability.shape[-1] >= size[1]:
            return F.adaptive_max_pool2d(probability, size)
        return F.interpolate(probability, size=size, mode='bilinear',
                             align_corners=False)

    def build_prior(self, probability):
        """Return peripheral-response and soft radial-envelope maps."""
        b, _, h, w = probability.shape
        device = probability.device
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32) + 0.5,
            torch.arange(w, device=device, dtype=torch.float32) + 0.5,
            indexing='ij')
        xx = xx.view(1, 1, h, w)
        yy = yy.view(1, 1, h, w)


        centroid_kernel = 15
        centroid_padding = centroid_kernel // 2
        mass = F.avg_pool2d(probability, centroid_kernel, stride=1,
                            padding=centroid_padding).clamp_min(1e-6)
        cx = F.avg_pool2d(probability * xx, centroid_kernel, stride=1,
                          padding=centroid_padding) / mass
        cy = F.avg_pool2d(probability * yy, centroid_kernel, stride=1,
                          padding=centroid_padding) / mass
        dx, dy = xx - cx, yy - cy
        radius = torch.sqrt(dx.square() + dy.square() + 1e-6)
        radius_norm = (radius / (centroid_kernel / 2.0)).clamp(max=2.0)

        local_density = F.avg_pool2d(probability, kernel_size=5, stride=1,
                                     padding=2)
        isolation = probability / local_density.clamp_min(1e-4)
        isolation = isolation / isolation.amax(dim=(2, 3), keepdim=True).clamp_min(1.0)


        confidence = probability * torch.sigmoid(
            (probability - self.point_threshold) / self.temperature)
        peripheral = confidence * (1.0 + self.radial_gain * radius_norm) * (
            1.0 + self.isolation_gain * isolation)
        peripheral = peripheral / peripheral.amax(
            dim=(2, 3), keepdim=True).clamp_min(1e-6)


        envelope = (
            0.50 * F.max_pool2d(peripheral, 3, stride=1, padding=1) +
            0.30 * F.max_pool2d(peripheral, 7, stride=1, padding=3) +
            0.20 * F.max_pool2d(peripheral, 15, stride=1, padding=7))
        envelope = F.avg_pool2d(envelope, 3, stride=1, padding=1).clamp(0.0, 1.0)
        reliability = F.max_pool2d(probability, 15, stride=1, padding=7)
        envelope = envelope * reliability.clamp(0.0, 1.0)
        return peripheral, envelope

    def forward(self, x):
        point_logits = GLOBAL_POINT_DICT.get(x.device, None)
        if point_logits is None:
            return x
        if point_logits.shape[0] != x.shape[0]:
            if x.shape[0] == 1:
                point_logits = point_logits[-1:]
            else:
                return x

        probability = self._resize_probability(point_logits, x.shape[-2:])
        peripheral, envelope = self.build_prior(probability)
        prior = torch.cat((peripheral, envelope), dim=1).to(dtype=x.dtype)
        point_feature = self.point_embed(prior)
        delta = self.feature_delta(torch.cat((x, point_feature), dim=1))
        gain = self.max_gain * torch.sigmoid(self.gain)
        return x + gain * delta


class OutwardPointFourierFusion(nn.Module):
    """One-way peripheral point evidence for a Fourier feature level.

    The point teacher is detached.  Local annular responses retain isolated
    outer members of a point set, while signed spatial derivatives encode the
    direction in which the Fourier feature should change.  The residual is
    positive-only and spatially gated, so this module cannot suppress the
    original Fourier feature or send Fourier gradients back into the point
    head.
    """

    def __init__(self, c1, max_gain=0.35, detach_prior=True, power=1.5,
                 point_threshold=0.12, temperature=0.04,
                 gain_init=-1.0, point_channels=32,
                 inner_kernel=3, middle_kernel=7, outer_kernel=15):
        super().__init__()
        self.c1 = int(c1)
        self.max_gain = max(float(max_gain), 0.0)
        self.detach_prior = bool(detach_prior)
        self.power = max(float(power), 1.0)
        self.point_threshold = float(point_threshold)
        self.temperature = max(float(temperature), 1e-3)
        self.point_channels = int(point_channels)
        kernels = [int(inner_kernel), int(middle_kernel), int(outer_kernel)]
        kernels = [max(value, 1) for value in kernels]
        kernels = [value if value % 2 else value + 1 for value in kernels]
        self.inner_kernel = kernels[0]
        self.middle_kernel = max(kernels[1], self.inner_kernel)
        self.outer_kernel = max(kernels[2], self.middle_kernel)
        hidden = max(self.c1 // 8, 16)
        groups = min(8, hidden)
        while hidden % groups:
            groups -= 1
        self.prior_embed = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.point_feature_proj = nn.Sequential(
            nn.Conv2d(self.point_channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.feature_delta = nn.Sequential(
            nn.Conv2d(self.c1 + hidden, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden,
                      bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c1, 1, bias=False),
        )
        self.gain_logit = nn.Parameter(torch.full((1,), float(gain_init)))
        self.last_stats = {}

    def _point_probability(self, x):
        point_logits = GLOBAL_POINT_DICT.get(x.device)
        if not torch.is_tensor(point_logits):
            return None
        if point_logits.shape[0] != x.shape[0]:
            if x.shape[0] == 1:
                point_logits = point_logits[-1:]
            else:
                return None
        probability = point_logits.sigmoid().float().pow(self.power)
        if self.detach_prior:
            probability = probability.detach()
        if probability.shape[-2:] != x.shape[-2:]:
            if (probability.shape[-2] >= x.shape[-2] and
                    probability.shape[-1] >= x.shape[-1]):
                probability = F.adaptive_max_pool2d(
                    probability, x.shape[-2:])
            else:
                probability = F.interpolate(
                    probability, size=x.shape[-2:], mode='bilinear',
                    align_corners=False)
        return probability

    def forward(self, x):
        probability = self._point_probability(x)
        if probability is None:
            self.last_debug = {}
            return x
        confidence = probability * torch.sigmoid(
            (probability - self.point_threshold) / self.temperature)


        inner_kernel = int(getattr(self, 'inner_kernel', 3))
        middle_kernel = int(getattr(self, 'middle_kernel', 7))
        outer_kernel = int(getattr(self, 'outer_kernel', 15))
        inner = F.max_pool2d(
            confidence, inner_kernel, stride=1,
            padding=inner_kernel // 2)
        middle = F.max_pool2d(
            confidence, middle_kernel, stride=1,
            padding=middle_kernel // 2)
        outer = F.max_pool2d(
            confidence, outer_kernel, stride=1,
            padding=outer_kernel // 2)
        middle_ring = F.relu(middle - inner)
        outer_ring = F.relu(outer - middle)


        grad_x = F.pad(confidence, (1, 1, 0, 0), mode='replicate')
        grad_x = 0.5 * (grad_x[..., 2:] - grad_x[..., :-2])
        grad_y = F.pad(confidence, (0, 0, 1, 1), mode='replicate')
        grad_y = 0.5 * (grad_y[..., 2:, :] - grad_y[..., :-2, :])
        prior = torch.cat(
            (confidence, middle_ring, outer_ring, grad_x, grad_y), dim=1
        ).to(dtype=x.dtype)
        embedded = self.prior_embed(prior)
        point_entry = GLOBAL_POINT_FEATURE_DICT.get(x.device)
        point_feature = (
            point_entry.get('feature') if isinstance(point_entry, dict)
            else None)
        if (torch.is_tensor(point_feature) and
                point_feature.shape[1] == self.point_channels):
            if point_feature.shape[0] != x.shape[0]:
                point_feature = (
                    point_feature[-1:] if x.shape[0] == 1 else None)
            if point_feature is not None:
                if self.detach_prior:
                    point_feature = point_feature.detach()
                point_feature = F.interpolate(
                    point_feature, size=x.shape[-2:], mode='bilinear',
                    align_corners=False).to(dtype=x.dtype)
                point_context = self.point_feature_proj(point_feature)
                external_mask = torch.maximum(
                    middle_ring, outer_ring).to(dtype=x.dtype)
                embedded = embedded + external_mask * point_context
        delta = F.relu(self.feature_delta(torch.cat((x, embedded), dim=1)))
        external_gate = torch.maximum(middle_ring, outer_ring)
        external_gate = torch.maximum(external_gate, confidence)
        gate = F.avg_pool2d(
            external_gate, 3, stride=1, padding=1).clamp(0.0, 1.0)
        gain = self.max_gain * torch.sigmoid(self.gain_logit)
        output = x + gain.to(x.dtype) * gate.to(x.dtype) * delta
        self.last_debug = {
            'point_probability': probability.detach(),
            'confidence': confidence.detach(),
            'middle_ring': middle_ring.detach(),
            'outer_ring': outer_ring.detach(),
            'grad_x': grad_x.detach(),
            'grad_y': grad_y.detach(),
            'gate': gate.detach(),
            'feature_in_energy': x.detach().float().abs().mean(
                dim=1, keepdim=True),
            'delta_energy': delta.detach().float().abs().mean(
                dim=1, keepdim=True),
            'feature_out_energy': output.detach().float().abs().mean(
                dim=1, keepdim=True),
        }
        self.last_stats = {
            'gain': gain.detach(),
            'external_mean': external_gate.detach().mean(),
            'external_max': external_gate.detach().amax(),
            'external_active': (
                external_gate.detach() > self.point_threshold).float().mean(),
        }
        return output


class CandidateDirectionalFourierBridge(OutwardPointFourierFusion):
    """Candidate-aware point support for differentiable Fourier reconstruction.

    The learned residual path is retained for backward-compatible feature
    conditioning. In addition, this bridge publishes a detached per-resolution
    point-support map together with the matching FourierProposal logits.
    FTInfer then samples that map around each selected proposal, computes its
    directional outer envelope and projects the envelope into Fourier
    coefficients with a fixed differentiable DFT.
    """

    def __init__(self, c1, max_gain=0.35, detach_prior=True, power=1.5,
                 point_threshold=0.12, temperature=0.04,
                 gain_init=-1.0, point_channels=32,
                 inner_kernel=3, middle_kernel=7, outer_kernel=15,
                 point_rescue=False, rescue_gain=0.85,
                 rescue_threshold=0.10, cluster_kernel=7,
                 cluster_min_mass=1.5, cluster_temperature=0.5,
                 cluster_nms_kernel=7, cluster_topk=4,
                 center_kernel=9, center_assignment_kernel=7,
                 preferred_anchor=0, hybrid_match=False,
                 hybrid_builder=False, hybrid_fourier_weight=1.00,
                 hybrid_scale_temperature=0.70,
                 hybrid_match_radius=0.05, hybrid_min_fourier=0.01,
                 hybrid_extent_gain=1.60,
                 hybrid_unmatched_fourier_threshold=1.10):
        super().__init__(
            c1, max_gain=max_gain, detach_prior=detach_prior, power=power,
            point_threshold=point_threshold, temperature=temperature,
            gain_init=gain_init, point_channels=point_channels,
            inner_kernel=inner_kernel, middle_kernel=middle_kernel,
            outer_kernel=outer_kernel)
        self.point_rescue = bool(point_rescue)
        self.rescue_gain = min(max(float(rescue_gain), 0.0), 1.0)
        self.rescue_threshold = min(
            max(float(rescue_threshold), 0.0), 1.0)
        self.cluster_kernel = self._odd_kernel(cluster_kernel)
        self.cluster_min_mass = max(float(cluster_min_mass), 0.0)
        self.cluster_temperature = max(float(cluster_temperature), 1e-3)
        self.cluster_nms_kernel = self._odd_kernel(cluster_nms_kernel)
        self.cluster_topk = max(int(cluster_topk), 0)
        self.center_kernel = self._odd_kernel(center_kernel)
        self.center_assignment_kernel = self._odd_kernel(
            center_assignment_kernel)
        self.preferred_anchor = int(preferred_anchor)
        self.hybrid_match = bool(hybrid_match)
        self.hybrid_builder = bool(hybrid_builder)
        self.hybrid_fourier_weight = min(max(
            float(hybrid_fourier_weight), 0.0), 1.0)
        self.hybrid_scale_temperature = max(
            float(hybrid_scale_temperature), 1e-3)
        self.hybrid_match_radius = max(float(hybrid_match_radius), 1e-4)
        self.hybrid_min_fourier = min(max(
            float(hybrid_min_fourier), 0.0), 1.0)
        self.hybrid_extent_gain = max(float(hybrid_extent_gain), 1.0)
        self.hybrid_unmatched_fourier_threshold = float(
            hybrid_unmatched_fourier_threshold)

    @staticmethod
    def _odd_kernel(value):
        value = max(int(value), 1)
        return value if value % 2 else value + 1

    @staticmethod
    def _patch_proposal_decoded(proposal_entry, probability):
        decoded = (
            proposal_entry.get('decoded')
            if isinstance(proposal_entry, dict) else None)
        if (torch.is_tensor(decoded) and
                decoded.shape[:4] == probability.shape and
                decoded.shape[-1] >= 5):
            with torch.no_grad():
                decoded[..., 4].copy_(
                    probability.to(dtype=decoded.dtype))

    def _publish_hybrid_arbitration(
            self, proposal_store, support_prob, source_resolution,
            rescue_debug):
        """Select one Fourier scale/anchor for every accepted point cluster.

        FourierProposal remains responsible for the candidate score and helps
        select the pyramid level/anchor. The point set supplies association,
        an extent prior and a center prior that FTInfer applies as a soft
        correction. All unmatched duplicate
        candidates are suppressed before NMS, while FTInfer still predicts the
        base Fourier coefficients from image features.
        """
        if not isinstance(proposal_store, dict):
            return
        accepted = rescue_debug.get('accepted_centers')
        center_x = rescue_debug.get('center_x')
        center_y = rescue_debug.get('center_y')
        if (not torch.is_tensor(accepted) or
                not torch.is_tensor(center_x) or
                not torch.is_tensor(center_y)):
            return

        level_items = sorted(
            ((resolution, entry)
             for resolution, entry in proposal_store.items()
             if isinstance(resolution, tuple) and isinstance(entry, dict) and
             torch.is_tensor(entry.get('logits'))),
            key=lambda item: int(item[1].get('level', 0)))
        if not level_items:
            return
        source_entry = proposal_store.get(source_resolution)
        if not isinstance(source_entry, dict):
            return
        source_stride = float(source_entry.get('stride', 1.0))
        source_h, source_w = source_resolution
        threshold = float(getattr(self, 'rescue_threshold', 0.10))
        active = (
            (support_prob.float() - threshold) /
            max(1.0 - threshold, 1e-3)
        ).clamp(0.0, 1.0)
        extent_radius = int(getattr(self, 'center_kernel', 9)) // 2
        extent_gain = float(getattr(self, 'hybrid_extent_gain', 1.60))
        fourier_weight = float(getattr(
            self, 'hybrid_fourier_weight', 1.00))
        scale_temperature = float(getattr(
            self, 'hybrid_scale_temperature', 0.70))
        match_radius = float(getattr(self, 'hybrid_match_radius', 0.05))
        min_fourier = float(getattr(self, 'hybrid_min_fourier', 0.01))
        unmatched_threshold = float(getattr(
            self, 'hybrid_unmatched_fourier_threshold', 1.10))
        rescue_gain = float(getattr(self, 'rescue_gain', 0.85))

        outputs = {}
        for resolution, entry in level_items:
            logits = entry['logits']
            work = logits[..., 0] if logits.ndim == 5 else logits
            batch, anchors_n, height, width = work.shape
            original = work.detach().float().sigmoid()


            if unmatched_threshold <= 1.0:
                hybrid_prob = torch.where(
                    original >= unmatched_threshold,
                    original, torch.full_like(original, 1e-5))
            else:
                hybrid_prob = torch.full_like(original, 1e-5)
            center_override = work.new_full(
                (batch, anchors_n, height, width, 2),
                float('nan'), dtype=torch.float32)
            point_candidate = torch.zeros_like(original)
            selected_scale = torch.zeros_like(original)
            outputs[resolution] = {
                'original': original,
                'prob': hybrid_prob,
                'center': center_override,
                'point': point_candidate,
                'scale': selected_scale,
            }

        batch_n = accepted.shape[0]
        cluster_records = []
        for batch_index in range(batch_n):
            locations = torch.nonzero(
                accepted[batch_index, 0] > 0.0, as_tuple=False)
            for location in locations:
                grid_y, grid_x = (int(location[0]), int(location[1]))
                point_score = accepted[
                    batch_index, 0, grid_y, grid_x].float()
                point_center_source = torch.stack((
                    center_x[batch_index, 0, grid_y, grid_x],
                    center_y[batch_index, 0, grid_y, grid_x]))
                center_pixels = point_center_source * source_stride

                y0 = max(grid_y - extent_radius, 0)
                y1 = min(grid_y + extent_radius + 1, source_h)
                x0 = max(grid_x - extent_radius, 0)
                x1 = min(grid_x + extent_radius + 1, source_w)
                local_active = active[
                    batch_index, 0, y0:y1, x0:x1]
                members = torch.nonzero(
                    local_active > 0.05, as_tuple=False)
                if members.numel():
                    member_h = (
                        members[:, 0].max() - members[:, 0].min() + 1
                    ).float()
                    member_w = (
                        members[:, 1].max() - members[:, 1].min() + 1
                    ).float()
                    extent_pixels = torch.stack((
                        member_w, member_h)) * source_stride * extent_gain
                else:
                    extent_pixels = point_center_source.new_tensor(
                        [source_stride, source_stride]) * extent_gain
                extent_pixels = extent_pixels.clamp_min(source_stride)

                level_records = []
                for resolution, entry in level_items:
                    state = outputs[resolution]
                    original = state['original']
                    if batch_index >= original.shape[0]:
                        continue
                    _, anchors_n, height, width = original.shape
                    stride = float(entry.get('stride', 1.0))
                    center_level = center_pixels / stride
                    radius_x = max(
                        int(round(match_radius * width)), 1)
                    radius_y = max(
                        int(round(match_radius * height)), 1)
                    center_x_value, center_y_value = (
                        center_level.detach().tolist())
                    base_x = min(max(
                        int(center_x_value), 0), width - 1)
                    base_y = min(max(
                        int(center_y_value), 0), height - 1)
                    lx0, lx1 = max(base_x - radius_x, 0), min(
                        base_x + radius_x + 1, width)
                    ly0, ly1 = max(base_y - radius_y, 0), min(
                        base_y + radius_y + 1, height)
                    anchor_pixels = entry.get('anchor_pixels')
                    if not torch.is_tensor(anchor_pixels):
                        continue
                    anchor_pixels = anchor_pixels.to(
                        device=original.device, dtype=torch.float32)
                    ratios = torch.maximum(
                        extent_pixels[None, :].to(anchor_pixels.device) /
                        anchor_pixels.clamp_min(1e-4),
                        anchor_pixels /
                        extent_pixels[None, :].to(anchor_pixels.device)
                        .clamp_min(1e-4)).amax(dim=-1)
                    scale_fit = torch.exp(
                        -0.5 * (
                            ratios.clamp_min(1.0).log() /
                            scale_temperature) ** 2)


                    patch = original[
                        batch_index, :, ly0:ly1, lx0:lx1]
                    patch_h, patch_w = patch.shape[-2:]
                    yy = torch.arange(
                        ly0, ly1, device=original.device,
                        dtype=torch.float32) + 0.5
                    xx = torch.arange(
                        lx0, lx1, device=original.device,
                        dtype=torch.float32) + 0.5
                    grid_y, grid_x = torch.meshgrid(
                        yy, xx, indexing='ij')
                    distance_norm = torch.sqrt(
                        ((grid_x - center_level[0]) /
                         max(width, 1)).square() +
                        ((grid_y - center_level[1]) /
                         max(height, 1)).square())
                    distance_fit = torch.exp(
                        -0.5 * (
                            distance_norm / match_radius).square())
                    selection = (
                        patch.clamp_min(1e-5).pow(fourier_weight) *
                        scale_fit[:, None, None].clamp_min(1e-5).pow(
                            1.0 - fourier_weight) *
                        distance_fit[None])
                    flat_index = selection.reshape(-1).argmax()
                    area = patch_h * patch_w
                    anchor_index = torch.div(
                        flat_index, area, rounding_mode='floor')
                    remainder = flat_index.remainder(area)
                    patch_y = torch.div(
                        remainder, patch_w, rounding_mode='floor')
                    patch_x = remainder.remainder(patch_w)
                    level_records.append((
                        selection.reshape(-1)[flat_index],
                        patch.reshape(-1)[flat_index],
                        resolution, anchor_index,
                        patch_y + ly0, patch_x + lx0,
                        scale_fit[anchor_index], center_level))

                if not level_records:
                    continue
                best_level = int(torch.stack([
                    record[0] for record in level_records
                ]).argmax().item())
                _, fourier_value, resolution, anchor_tensor, candidate_y_tensor,\
                    candidate_x_tensor, scale_fit, center_level =\
                    level_records[best_level]
                anchor_index = int(anchor_tensor.item())
                candidate_y = int(candidate_y_tensor.item())
                candidate_x = int(candidate_x_tensor.item())
                state = outputs[resolution]
                original_value = state['original'][
                    batch_index, anchor_index, candidate_y, candidate_x]


                point_boost = (
                    rescue_gain * point_score * scale_fit
                ).clamp(0.0, 1.0)
                point_boost = point_boost * torch.where(
                    fourier_value < min_fourier,
                    1.0 - min_fourier + fourier_value,
                    fourier_value.new_ones(()))
                joint_value = (
                    1.0 - (1.0 - original_value) *
                    (1.0 - point_boost)
                ).clamp(1e-5, 1.0 - 1e-5)
                state['prob'][
                    batch_index, anchor_index, candidate_y, candidate_x
                ] = torch.maximum(
                    state['prob'][
                        batch_index, anchor_index, candidate_y, candidate_x],
                    joint_value)
                state['point'][
                    batch_index, anchor_index, candidate_y, candidate_x
                ] = point_boost
                state['scale'][
                    batch_index, anchor_index, candidate_y, candidate_x
                ] = int(proposal_store[resolution].get('level', 0)) + 1
                state['center'][
                    batch_index, anchor_index, candidate_y, candidate_x
                ] = center_level
                cluster_records.append({
                    'batch': batch_index,
                    'point_score': point_score.detach(),
                    'extent_pixels': extent_pixels.detach(),
                    'center_pixels': center_pixels.detach(),
                    'level': int(
                        proposal_store[resolution].get('level', 0)),
                    'anchor': anchor_index,
                    'fourier_score': original_value.detach(),
                    'joint_score': joint_value.detach(),
                })

        for resolution, entry in level_items:
            state = outputs[resolution]
            probability = state['prob'].clamp(1e-5, 1.0 - 1e-5)
            entry['hybrid_logits'] = torch.logit(
                probability).unsqueeze(-1)
            entry['hybrid_prob'] = probability
            entry['hybrid_center_override'] = state['center']
            entry['hybrid_debug'] = {
                'fourier_candidate_prob': state['original'].detach(),
                'point_candidate_prob': state['point'].detach(),
                'joint_candidate_prob': probability.detach(),
                'hybrid_selected_scale': state['scale'].detach(),
                'center_valid': torch.isfinite(
                    state['center']).all(dim=-1).float().detach(),
            }
            self._patch_proposal_decoded(entry, probability)
        proposal_store['_hybrid_ready'] = True
        proposal_store['_hybrid_records'] = cluster_records

    def _point_conditioned_candidates(
            self, support_prob, candidate_logits, proposal_entry,
            proposal_store=None, resolution=None):
        """Build sparse point-set candidates and their local centroid field.

        Point responses are first converted into cluster evidence using local
        mass, which rejects an isolated high response. Spatial NMS then leaves
        only a few formation candidates. The original Fourier probability and
        the point candidate probability are combined by a soft union. A local
        point-weighted centroid is published in feature coordinates so FTInfer
        can correct the Fourier center before directional reconstruction.
        """
        hybrid_match = bool(getattr(self, 'hybrid_match', False))
        hybrid_builder = bool(getattr(self, 'hybrid_builder', False))
        if (hybrid_match and not hybrid_builder and
                isinstance(proposal_entry, dict) and
                torch.is_tensor(proposal_entry.get('hybrid_logits'))):
            return (
                proposal_entry['hybrid_logits'],
                proposal_entry.get('hybrid_center_override'),
                proposal_entry.get('hybrid_debug', {}))
        if (not bool(getattr(self, 'point_rescue', False)) or
                not torch.is_tensor(candidate_logits) or
                int(getattr(self, 'cluster_topk', 0)) <= 0):
            return candidate_logits, None, {}

        if candidate_logits.ndim == 5 and candidate_logits.shape[-1] == 1:
            candidate_work = candidate_logits[..., 0]
        else:
            candidate_work = candidate_logits
        if candidate_work.ndim != 4:
            return candidate_logits, None, {}

        batch, anchors_n, height, width = candidate_work.shape
        probability = support_prob.float().clamp(0.0, 1.0)
        threshold = float(getattr(self, 'rescue_threshold', 0.10))
        active = (
            (probability - threshold) /
            max(1.0 - threshold, 1e-3)
        ).clamp(0.0, 1.0)
        cluster_kernel = int(getattr(self, 'cluster_kernel', 7))
        local_mass = F.avg_pool2d(
            active, cluster_kernel, stride=1,
            padding=cluster_kernel // 2) * float(cluster_kernel ** 2)
        local_peak = F.max_pool2d(
            probability, cluster_kernel, stride=1,
            padding=cluster_kernel // 2)
        density = torch.sigmoid(
            (local_mass - float(getattr(
                self, 'cluster_min_mass', 1.5))) /
            max(float(getattr(
                self, 'cluster_temperature', 0.5)), 1e-3))
        cluster_score = (local_peak * density).clamp(0.0, 1.0)

        nms_kernel = int(getattr(self, 'cluster_nms_kernel', 7))


        yy, xx = torch.meshgrid(
            torch.arange(
                height, device=probability.device, dtype=torch.float32) + 0.5,
            torch.arange(
                width, device=probability.device, dtype=torch.float32) + 0.5,
            indexing='ij')
        center_kernel = int(getattr(self, 'center_kernel', 9))
        center_area = float(center_kernel ** 2)
        center_weight = active * probability
        weight_sum = F.avg_pool2d(
            center_weight, center_kernel, stride=1,
            padding=center_kernel // 2) * center_area
        center_x = F.avg_pool2d(
            center_weight * xx.view(1, 1, height, width),
            center_kernel, stride=1,
            padding=center_kernel // 2) * center_area
        center_y = F.avg_pool2d(
            center_weight * yy.view(1, 1, height, width),
            center_kernel, stride=1,
            padding=center_kernel // 2) * center_area
        centroid = torch.stack((
            center_x / weight_sum.clamp_min(1e-6),
            center_y / weight_sum.clamp_min(1e-6)), dim=-1)

        rescue = candidate_work.new_zeros(
            (batch, anchors_n, height, width), dtype=torch.float32)
        center_override = candidate_work.new_full(
            (batch, anchors_n, height, width, 2),
            float('nan'), dtype=torch.float32)
        assignment_radius = int(getattr(
            self, 'center_assignment_kernel', 7)) // 2
        topk = min(int(getattr(self, 'cluster_topk', 4)), height * width)
        preferred_anchor = min(max(
            int(getattr(self, 'preferred_anchor', 0)), 0), anchors_n - 1)
        accepted = candidate_work.new_zeros(
            (batch, 1, height, width), dtype=torch.float32)

        for batch_index in range(batch):


            work_score = cluster_score[batch_index, 0].clone()
            for rank in range(topk):
                score, flat_tensor = work_score.reshape(-1).max(dim=0)
                if float(score.detach()) < threshold:
                    continue
                flat_index = int(flat_tensor.item())
                grid_y, grid_x = divmod(flat_index, width)
                point_center = centroid[batch_index, 0, grid_y, grid_x]
                if (not torch.isfinite(point_center).all() or
                        float(weight_sum[
                            batch_index, 0, grid_y, grid_x].detach()) <= 0.0):
                    continue
                rescue[
                    batch_index, preferred_anchor, grid_y, grid_x
                ] = score
                accepted[batch_index, 0, grid_y, grid_x] = score
                y0 = max(grid_y - assignment_radius, 0)
                y1 = min(grid_y + assignment_radius + 1, height)
                x0 = max(grid_x - assignment_radius, 0)
                x1 = min(grid_x + assignment_radius + 1, width)
                center_override[
                    batch_index, :, y0:y1, x0:x1
                ] = point_center
                suppress_radius = nms_kernel // 2
                sy0 = max(grid_y - suppress_radius, 0)
                sy1 = min(grid_y + suppress_radius + 1, height)
                sx0 = max(grid_x - suppress_radius, 0)
                sx1 = min(grid_x + suppress_radius + 1, width)
                work_score[sy0:sy1, sx0:sx1] = 0.0

        original_probability = candidate_work.detach().float().sigmoid()
        point_probability = (
            float(getattr(self, 'rescue_gain', 0.85)) * rescue
        ).clamp(0.0, 1.0)
        joint_probability = (
            1.0 -
            (1.0 - original_probability) * (1.0 - point_probability)
        ).clamp(1e-5, 1.0 - 1e-5)
        joint_logits = torch.logit(joint_probability).unsqueeze(-1)
        grid_centers = torch.stack((xx, yy), dim=-1).view(
            1, 1, height, width, 2)
        center_valid = torch.isfinite(center_override).all(
            dim=-1, keepdim=False)
        center_shift = torch.linalg.vector_norm(
            torch.nan_to_num(center_override - grid_centers), dim=-1)


        self._patch_proposal_decoded(proposal_entry, joint_probability)

        rescue_debug = {
            'cluster_mass': local_mass.detach(),
            'cluster_score': cluster_score.detach(),
            'point_candidate_prob': point_probability.detach(),
            'joint_candidate_prob': joint_probability.detach(),
            'accepted_centers': accepted.detach(),
            'center_x': centroid[..., 0].detach(),
            'center_y': centroid[..., 1].detach(),
            'center_valid': center_valid.detach().float(),
            'center_shift': center_shift.detach(),
        }
        if (hybrid_match and hybrid_builder):
            self._publish_hybrid_arbitration(
                proposal_store, support_prob, resolution, rescue_debug)
            if (isinstance(proposal_entry, dict) and
                    torch.is_tensor(proposal_entry.get('hybrid_logits'))):
                rescue_debug.update(
                    proposal_entry.get('hybrid_debug', {}))
                return (
                    proposal_entry['hybrid_logits'],
                    proposal_entry.get('hybrid_center_override'),
                    rescue_debug)
        return joint_logits, center_override, rescue_debug

    def forward(self, x):
        output = super().forward(x)
        debug = getattr(self, 'last_debug', {})
        support_prob = debug.get('point_probability')
        if not torch.is_tensor(support_prob):
            return output

        resolution = tuple(int(value) for value in x.shape[-2:])
        proposal_store = GLOBAL_FOURIER_PROPOSAL_DICT.get(x.device)
        proposal_entry = (
            proposal_store.get(resolution)
            if isinstance(proposal_store, dict) else None)
        candidate_logits = (
            proposal_entry.get('logits')
            if isinstance(proposal_entry, dict) else None)
        if (torch.is_tensor(candidate_logits) and
                candidate_logits.shape[0] != x.shape[0]):
            candidate_logits = (
                candidate_logits[-x.shape[0]:]
                if candidate_logits.shape[0] >= x.shape[0] else None)

        support_prob = support_prob.float().clamp(1e-5, 1.0 - 1e-5)
        candidate_logits, center_override, rescue_debug =\
            self._point_conditioned_candidates(
                support_prob, candidate_logits, proposal_entry,
                proposal_store=proposal_store, resolution=resolution)
        if rescue_debug:
            debug.update(rescue_debug)
            if isinstance(proposal_entry, dict):
                proposal_entry['joint_logits'] = candidate_logits.detach()
                proposal_entry['joint_prob'] = (
                    candidate_logits.detach().float().sigmoid())
        support_store = GLOBAL_FORMATION_SUPPORT_DICT.get(x.device)
        if not isinstance(support_store, dict):
            support_store = {}
        support_store[resolution] = {
            'logits': torch.logit(support_prob),
            'prob': support_prob,
            'candidate_logits': (
                candidate_logits.detach()
                if torch.is_tensor(candidate_logits) else None),
            'center_override': (
                center_override.detach()
                if torch.is_tensor(center_override) else None),
            'learned_logits': None,
            'bridge_debug': debug,
        }
        GLOBAL_FORMATION_SUPPORT_DICT[x.device] = support_store

        geometry_store = GLOBAL_CANDIDATE_GEOMETRY_DICT.get(x.device)
        if not isinstance(geometry_store, dict):
            geometry_store = {}
        entry = geometry_store.get(resolution, {})
        entry.update({
            'support_prob': support_prob.detach(),
            'candidate_prob': (
                candidate_logits.detach().float().sigmoid()
                if torch.is_tensor(candidate_logits) else None),
            'point_candidate_prob': rescue_debug.get(
                'point_candidate_prob'),
            'joint_candidate_prob': rescue_debug.get(
                'joint_candidate_prob'),
            'center_override': (
                center_override.detach()
                if torch.is_tensor(center_override) else None),
            'bridge_debug': debug,
        })
        geometry_store[resolution] = entry
        GLOBAL_CANDIDATE_GEOMETRY_DICT[x.device] = geometry_store
        return output


class YoloPointFourierFeatureBridge(nn.Module):
    """Trainable YOLO/point feature injection for one Fourier pyramid level.

    Detect caches its pre-prediction FPN feature maps and BioCountInfer caches
    its decoded point feature.  This module retrieves the resolution-matched
    YOLO feature, resizes the point feature, encodes multi-dilation formation
    context, and injects both through a spatial gate.  The output remains a
    Fourier feature tensor; no box or contour is constructed in post-process.

    Detaching the two teachers is the stable Stage-B setting: Fourier losses
    train this bridge and the Fourier branch without corrupting the already
    trained YOLO/point heads.  A later joint stage may disable either detach.
    """

    def __init__(self, c1, point_channels=32, max_point_gain=0.25,
                 max_yolo_gain=0.10, detach_point=True, detach_yolo=True,
                 point_gain_init=-2.0, yolo_gain_init=-3.0,
                 point_grad_scale=0.10, yolo_grad_scale=0.05):
        super().__init__()
        self.c1 = int(c1)
        self.point_channels = int(point_channels)
        self.max_point_gain = float(max_point_gain)
        self.max_yolo_gain = float(max_yolo_gain)
        self.detach_point = bool(detach_point)
        self.detach_yolo = bool(detach_yolo)
        self.point_grad_scale = float(point_grad_scale)
        self.yolo_grad_scale = float(yolo_grad_scale)
        hidden = max(self.c1 // 4, 32)

        self.point_proj = nn.Sequential(
            nn.Conv2d(self.point_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )


        self.point_context = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=dilation,
                          dilation=dilation, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ) for dilation in (1, 3, 5)
        ])
        self.yolo_proj = nn.Sequential(
            nn.Conv2d(self.c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.c1 + hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c1, 1),
            nn.Sigmoid(),
        )
        self.point_delta = nn.Sequential(
            nn.Conv2d(hidden, self.c1, 1, bias=False),
            nn.BatchNorm2d(self.c1),
        )
        self.yolo_delta = nn.Sequential(
            nn.Conv2d(hidden, self.c1, 1, bias=False),
            nn.BatchNorm2d(self.c1),
        )
        self.point_gain = nn.Parameter(torch.full((1,), float(point_gain_init)))
        self.yolo_gain = nn.Parameter(torch.full((1,), float(yolo_gain_init)))
        self.last_stats = {}

    @staticmethod
    def _match_batch(feature, reference):
        if feature.shape[0] == reference.shape[0]:
            return feature
        if reference.shape[0] == 1:
            return feature[-1:]
        return None

    def _get_yolo_feature(self, x):
        cached = GLOBAL_MACRO_DETECT_DICT.get(x.device)
        features = cached.get('features') if isinstance(cached, dict) else None
        if not isinstance(features, (list, tuple)):
            return None
        exact = [feature for feature in features
                 if torch.is_tensor(feature) and feature.ndim == 4 and
                 feature.shape[1] == self.c1 and
                 feature.shape[-2:] == x.shape[-2:]]
        return exact[0] if exact else None

    def forward(self, x):
        point_entry = GLOBAL_POINT_FEATURE_DICT.get(x.device)
        point_feature = (point_entry.get('feature')
                         if isinstance(point_entry, dict) else None)
        yolo_feature = self._get_yolo_feature(x)
        if not torch.is_tensor(point_feature) or not torch.is_tensor(yolo_feature):
            return x
        point_feature = self._match_batch(point_feature, x)
        yolo_feature = self._match_batch(yolo_feature, x)
        if point_feature is None or yolo_feature is None:
            return x
        if point_feature.shape[1] != self.point_channels:
            return x
        if self.detach_point:
            point_feature = point_feature.detach()
        elif self.point_grad_scale < 1.0:
            point_feature = (point_feature.detach() + self.point_grad_scale *
                             (point_feature - point_feature.detach()))
        if self.detach_yolo:
            yolo_feature = yolo_feature.detach()
        elif self.yolo_grad_scale < 1.0:
            yolo_feature = (yolo_feature.detach() + self.yolo_grad_scale *
                            (yolo_feature - yolo_feature.detach()))
        point_feature = F.interpolate(
            point_feature, size=x.shape[-2:], mode='bilinear',
            align_corners=False).to(dtype=x.dtype)
        yolo_feature = yolo_feature.to(dtype=x.dtype)

        point_base = self.point_proj(point_feature)
        point_context = torch.stack(
            [branch(point_base) for branch in self.point_context], dim=0).mean(dim=0)
        yolo_context = self.yolo_proj(yolo_feature)
        gate = self.spatial_gate(torch.cat((x, point_context, yolo_context), dim=1))
        point_gain = self.max_point_gain * torch.sigmoid(self.point_gain)
        yolo_gain = self.max_yolo_gain * torch.sigmoid(self.yolo_gain)
        output = x + gate * (
            point_gain * self.point_delta(point_context) +
            yolo_gain * self.yolo_delta(yolo_context))
        self.last_stats = {
            'point_gain': point_gain.detach(),
            'yolo_gain': yolo_gain.detach(),
            'gate_mean': gate.detach().mean(),
        }
        return output


class OraclePeripheralFourierBridge(nn.Module):
    """GT upper-bound bridge for peripheral point-to-Fourier interaction.

    Macro box GT defines a soft formation region and assigns each micro GT to
    its nearest normalized box.  Only peripheral, direction-preserving point
    evidence is encoded.  The bridge never sees Fourier GT coefficients and
    therefore cannot copy the target contour into the prediction.
    """

    def __init__(self, c1, max_gain=0.20, peripheral_start=0.45,
                 peripheral_temperature=0.10, box_scale=1.15,
                 gain_init_logit=-2.2):
        super().__init__()
        self.c1 = int(c1)
        self.max_gain = max(float(max_gain), 0.0)
        self.peripheral_start = float(peripheral_start)
        self.peripheral_temperature = max(float(peripheral_temperature), 1e-3)
        self.box_scale = max(float(box_scale), 1.0)
        hidden = max(self.c1 // 4, 24)
        self.encoder = nn.Sequential(
            nn.Conv2d(7, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(self.c1 + hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c1, 1),
            nn.Sigmoid(),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(hidden, self.c1, 1, bias=False),
            nn.BatchNorm2d(self.c1),
        )


        self.gain_logit = nn.Parameter(torch.tensor(float(gain_init_logit)))
        self.last_stats = {}

    @torch.no_grad()
    def _build_guidance(self, x, targets):
        batch, _, height, width = x.shape
        guide = torch.zeros((batch, 7, height, width), device=x.device,
                            dtype=torch.float32)
        if not torch.is_tensor(targets) or targets.numel() == 0:
            return guide
        targets = targets.to(device=x.device, dtype=torch.float32)
        macro = targets[targets[:, 1] == 1.0]
        micro = targets[targets[:, 1] == 0.0]
        if macro.numel() == 0:
            return guide

        yy = (torch.arange(height, device=x.device, dtype=torch.float32) + 0.5) / height
        xx = (torch.arange(width, device=x.device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        softness = 2.0 / max(height, width)

        for batch_idx in range(batch):
            macro_b = macro[macro[:, 0].long() == batch_idx]
            if macro_b.numel() == 0:
                continue

            for box in macro_b:
                cx, cy, bw, bh = box[2:6]
                half_w = 0.5 * bw * self.box_scale
                half_h = 0.5 * bh * self.box_scale
                mask_x = (torch.sigmoid((xx - (cx - half_w)) / softness) *
                          torch.sigmoid(((cx + half_w) - xx) / softness))
                mask_y = (torch.sigmoid((yy - (cy - half_h)) / softness) *
                          torch.sigmoid(((cy + half_h) - yy) / softness))
                guide[batch_idx, 6] = torch.maximum(
                    guide[batch_idx, 6], mask_x * mask_y)

            micro_b = micro[micro[:, 0].long() == batch_idx]
            if micro_b.numel() == 0:
                continue
            points = micro_b[:, 2:4]
            centers = macro_b[:, 2:4]
            half_size = (0.5 * macro_b[:, 4:6] * self.box_scale).clamp_min(1e-4)
            normalized = (points[:, None, :] - centers[None, :, :]) / half_size[None, :, :]
            normalized_radius = normalized.square().sum(dim=-1).sqrt()
            assignment = normalized_radius.argmin(dim=1)
            vectors = normalized[torch.arange(points.shape[0], device=x.device), assignment]
            radius = vectors.square().sum(dim=1).sqrt()
            direction = vectors / radius[:, None].clamp_min(1e-6)
            peripheral = torch.sigmoid(
                (radius - self.peripheral_start) / self.peripheral_temperature)

            gx = (points[:, 0] * width).long().clamp(0, width - 1)
            gy = (points[:, 1] * height).long().clamp(0, height - 1)
            b_index = torch.full_like(gx, batch_idx)
            values = torch.stack((
                peripheral,
                (peripheral * radius).clamp_max(2.0),
                peripheral * direction[:, 0].clamp_min(0.0),
                peripheral * (-direction[:, 0]).clamp_min(0.0),
                peripheral * direction[:, 1].clamp_min(0.0),
                peripheral * (-direction[:, 1]).clamp_min(0.0),
            ), dim=1)
            for channel in range(6):
                guide[:, channel].index_put_(
                    (b_index, gy, gx), values[:, channel], accumulate=True)


        guide[:, :6] = F.max_pool2d(
            guide[:, :6], kernel_size=3, stride=1, padding=1).clamp(0.0, 2.0)
        guide[:, :6] *= guide[:, 6:7]
        return guide

    def forward(self, x):
        oracle = GLOBAL_ORACLE_PERIPHERAL_DICT.get(x.device)
        targets = oracle.get('targets') if isinstance(oracle, dict) else None
        if not torch.is_tensor(targets):
            return x
        base_store = GLOBAL_ORACLE_FOURIER_BASE_DICT.get(x.device)
        if not isinstance(base_store, dict):
            base_store = {}
        base_store[tuple(x.shape[-2:])] = x
        GLOBAL_ORACLE_FOURIER_BASE_DICT[x.device] = base_store

        guidance = self._build_guidance(x, targets).to(dtype=x.dtype)
        if not bool((guidance[:, :6].amax() > 0).item()):
            return x
        encoded = self.encoder(guidance)
        gate = self.gate(torch.cat((x, encoded), dim=1))
        gain = self.max_gain * torch.sigmoid(self.gain_logit)
        output = x + gain * gate * self.delta(encoded)
        self.last_stats = {
            'gain': gain.detach(),
            'gate_mean': gate.detach().mean(),
            'peripheral_max': guidance[:, 0].detach().amax(),
            'peripheral_mean': guidance[:, 0].detach().mean(),
        }
        return output


class FourierPointHeatmapFusion(nn.Module):
    """Inject a sharpened micro-target heatmap into one Fourier feature level."""

    def __init__(self, c1, max_gain=0.10, power=4.0, detach_prior=True):
        super().__init__()
        self.c1 = c1
        self.max_gain = float(max_gain)
        self.power = float(power)
        self.detach_prior = bool(detach_prior)
        hidden = max(c1 // 8, 8)
        self.heat_embed = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 + hidden, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )


        self.gain = nn.Parameter(torch.full((1,), 0.01))

    def forward(self, x):
        point_logits = GLOBAL_POINT_DICT.get(x.device, None)
        if point_logits is None:
            return x
        if point_logits.shape[0] != x.shape[0]:
            if x.shape[0] == 1:
                point_logits = point_logits[-1:]
            else:
                return x

        point_prior = torch.sigmoid(point_logits.float()).pow(self.power)
        if self.detach_prior:
            point_prior = point_prior.detach()
        point_prior = F.interpolate(
            point_prior, size=x.shape[-2:], mode='bilinear', align_corners=False
        ).to(dtype=x.dtype)

        heat_feature = self.heat_embed(point_prior)
        delta = self.fusion(torch.cat((x * point_prior, heat_feature), dim=1))
        gain = self.max_gain * torch.tanh(self.gain)
        return x + gain * delta


class FormationSupportBridge(nn.Module):
    """Build one level of a differentiable formation-support pyramid.

    Point logits are detached deliberately so Fourier losses cannot corrupt the
    micro-target head. Max-pool dilation followed by min-pool erosion performs a
    differentiable morphological closing operation before learned feature fusion.
    """

    def __init__(self, c1, max_gain=0.10, kernel_small=5, kernel_medium=11,
                 kernel_large=21, detach_point=True, deterministic_envelope=False,
                 point_threshold=0.30, temperature=0.08):
        super().__init__()
        self.c1 = c1
        self.max_gain = float(max_gain)
        self.detach_point = bool(detach_point)
        self.deterministic_envelope = bool(deterministic_envelope)
        self.point_threshold = float(point_threshold)
        self.temperature = max(float(temperature), 1e-3)
        self.kernels = tuple(self._odd_kernel(k) for k in (kernel_small, kernel_medium, kernel_large))
        hidden = max(c1 // 4, 32)


        self.support_encoder = nn.Sequential(
            nn.Conv2d(c1 + 4, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.support_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.feature_gate = nn.Sequential(
            nn.Conv2d(c1 + 1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid(),
        )
        self.feature_delta = nn.Sequential(
            nn.Conv2d(c1 + hidden, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
        )

    @staticmethod
    def _odd_kernel(value):
        value = max(int(value), 1)
        return value if value % 2 else value + 1

    @staticmethod
    def _match_batch(point_logits, x):
        if point_logits.shape[0] == x.shape[0]:
            return point_logits
        if x.shape[0] == 1:
            return point_logits[-1:]
        return None

    @staticmethod
    def _close(probability, kernel_size):
        """Differentiable grayscale closing: dilation followed by erosion."""
        padding = kernel_size // 2
        dilated = F.max_pool2d(
            probability, kernel_size=kernel_size, stride=1, padding=padding)
        return -F.max_pool2d(
            -dilated, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        point_logits = GLOBAL_POINT_DICT.get(x.device, None)
        if point_logits is None:
            return x
        point_logits = self._match_batch(point_logits, x)
        if point_logits is None:
            return x

        point_prob = torch.sigmoid(point_logits.float())
        if self.detach_point:
            point_prob = point_prob.detach()
        point_prob = F.interpolate(point_prob, size=x.shape[-2:], mode='bilinear', align_corners=False).to(x.dtype)
        closed = [self._close(point_prob, k) for k in self.kernels]

        if getattr(self, 'deterministic_envelope', False):


            soft_points = torch.sigmoid(
                (point_prob - self.point_threshold) / self.temperature)
            closed_soft = [self._close(soft_points, k) for k in self.kernels]
            support_prob = torch.stack([soft_points, *closed_soft], dim=0).amax(dim=0)
            support_prob = support_prob.clamp(1e-4, 1.0 - 1e-4)
            support_logits = torch.logit(support_prob)


            if self.max_gain <= 0.0:
                support_store = GLOBAL_FORMATION_SUPPORT_DICT.get(x.device)
                if not isinstance(support_store, dict):
                    support_store = {}
                support_store[tuple(x.shape[-2:])] = {
                    'logits': support_logits,
                    'prob': support_prob,
                    'learned_logits': None,
                }
                GLOBAL_FORMATION_SUPPORT_DICT[x.device] = support_store
                return x
        else:
            support_feature = self.support_encoder(torch.cat((x, point_prob, *closed), dim=1))
            learned_support_logits = self.support_head(support_feature)
            support_logits = learned_support_logits
            support_prob = torch.sigmoid(support_logits)
        if getattr(self, 'deterministic_envelope', False):
            support_feature = self.support_encoder(torch.cat((x, point_prob, *closed), dim=1))
            learned_support_logits = self.support_head(support_feature)
        support_store = GLOBAL_FORMATION_SUPPORT_DICT.get(x.device)
        if not isinstance(support_store, dict):
            support_store = {}
        support_store[tuple(x.shape[-2:])] = {
            'logits': support_logits,
            'prob': support_prob,
            'learned_logits': learned_support_logits,
        }
        GLOBAL_FORMATION_SUPPORT_DICT[x.device] = support_store

        gate = self.feature_gate(torch.cat((x, support_prob), dim=1))
        delta = self.feature_delta(torch.cat((x * support_prob, support_feature), dim=1))
        return x + self.max_gain * gate * delta


class ASSCAModule(nn.Module):
    def __init__(self, c1, seq_len=5, gamma_scale=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.c1 = c1


        self.spatial_branch = SpatialGatingBranch(c1)
        self.freq_branch = AdvancedFrequencyBranch(c1)
        self.motion_branch = MotionConsistencyBranch(c1, seq_len)
        hidden = max(c1 // 4, 1)
        self.branch_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 * 4, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )


        self.fusion_mixer = nn.Conv2d(c1 * 4, c1, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(c1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma_scale = float(gamma_scale)
        self.vis_step_counter = 0


    def _tensor_to_heatmap(self, tensor, colormap=cv2.COLORMAP_JET, normalize=True):
        if tensor.dim() == 3:
            heatmap_data = tensor.mean(dim=0).detach().cpu().float().numpy()
        elif tensor.dim() == 2:
            heatmap_data = tensor.detach().cpu().float().numpy()
        else:
            raise ValueError(f"Unsupported tensor shape for visualization: {tensor.shape}")

        if normalize:
            min_val = heatmap_data.min()
            max_val = heatmap_data.max()
            heatmap_data = (heatmap_data - min_val) / (max_val - min_val + 1e-8)

        heatmap_uint8 = (np.clip(heatmap_data, 0, 1) * 255).astype(np.uint8)

        if colormap is not None:
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        else:
            heatmap_bgr = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2BGR)

        return heatmap_bgr


    def forward(self, x):
        bs_t, c, h, w = x.shape
        real_batch = bs_t // self.seq_len


        feat_s_all = self.spatial_branch(x).view(real_batch, self.seq_len, c, h, w)
        feat_f_all = self.freq_branch(x).view(real_batch, self.seq_len, c, h, w)


        feat_seq_flat = x.view(real_batch, self.seq_len * c, h, w)
        feat_m_global, motion_mask_5d = self.motion_branch(feat_seq_flat)


        x_seq = x.view(real_batch, self.seq_len, c, h, w)
        enhanced_seq = []


        memory = torch.zeros_like(x_seq[:, 0, ...])


        for t in range(self.seq_len):
            curr_s = torch.nan_to_num(feat_s_all[:, t, ...], nan=0.0, posinf=10.0, neginf=-10.0)
            curr_f = torch.nan_to_num(feat_f_all[:, t, ...], nan=0.0, posinf=10.0, neginf=-10.0)

            curr_attn_gate = torch.nan_to_num(motion_mask_5d[:, t, ...], nan=0.0, posinf=1.0, neginf=0.0)
            curr_attn_gate = torch.clamp(curr_attn_gate, 0.0, 1.0)
            curr_m = torch.nan_to_num(feat_m_global, nan=0.0, posinf=10.0, neginf=-10.0) * curr_attn_gate

            combined_raw = torch.cat([curr_s, curr_f, curr_m, memory], dim=1)
            if hasattr(self, 'branch_weight'):
                branch_w = torch.softmax(self.branch_weight(combined_raw), dim=1)
            else:
                branch_w = combined_raw.new_full((real_batch, 3, 1, 1), 1.0 / 3.0)
            motion_density = curr_attn_gate.mean(dim=(2, 3), keepdim=True)
            motion_reliability = torch.exp(-4.0 * motion_density).to(branch_w.dtype)
            branch_w = torch.cat([
                branch_w[:, 0:1],
                branch_w[:, 1:2],
                branch_w[:, 2:3] * motion_reliability,
            ], dim=1)
            branch_w = branch_w / (branch_w.sum(dim=1, keepdim=True) + 1e-6)

            curr_s = curr_s * branch_w[:, 0:1]
            curr_f = curr_f * branch_w[:, 1:2]
            curr_m = curr_m * branch_w[:, 2:3]
            combined = torch.cat([curr_s, curr_f, curr_m, memory], dim=1)

            fusion_out = self.bn_fusion(self.fusion_mixer(combined))
            fusion_out = torch.nan_to_num(fusion_out, nan=0.0, posinf=10.0, neginf=-10.0)
            delta_t = torch.clamp(fusion_out, -5.0, 5.0)

            memory = delta_t.detach()
            gamma_scale = getattr(self, 'gamma_scale', 0.1)
            gamma = gamma_scale * torch.tanh(self.gamma)
            feat_enhanced = x_seq[:, t, ...] + gamma * delta_t
            feat_enhanced = torch.nan_to_num(feat_enhanced, nan=0.0, posinf=10.0, neginf=-10.0)
            enhanced_seq.append(feat_enhanced)


            if self.training and t == self.seq_len - 1:
                self.vis_step_counter += 1
                if self.vis_step_counter % 100 == 0:
                    try:
                        w_vis = branch_w[0, :, 0, 0].detach().float().cpu().tolist()
                        img_bgr = self._tensor_to_heatmap(x_seq[0, t], colormap=None)
                        cv2.putText(img_bgr, f"Origin Input (T={t})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        spatial_map = self._tensor_to_heatmap(feat_s_all[0, t])
                        cv2.putText(spatial_map, f"Spatial (w:{w_vis[0]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        freq_map = self._tensor_to_heatmap(feat_f_all[0, t])
                        cv2.putText(freq_map, f"Freq (w:{w_vis[1]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        motion_map = self._tensor_to_heatmap(motion_mask_5d[0, t, 0])
                        cv2.putText(motion_map, f"Motion Mask (w:{w_vis[2]:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        final_map = self._tensor_to_heatmap(delta_t[0])
                        gamma_val = self.gamma.item()
                        cv2.putText(final_map, f"Fused (gamma:{gamma_val:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        overlay = cv2.addWeighted(img_bgr, 0.6, final_map, 0.4, 0)
                        cv2.putText(overlay, "Final Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        row1 = np.hstack([img_bgr, spatial_map, freq_map])
                        row2 = np.hstack([motion_map, final_map, overlay])
                        full_canvas = np.vstack([row1, row2])
                        full_canvas = cv2.resize(full_canvas, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

                        save_dir = 'debug_assca_probes'
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'step_{self.vis_step_counter:06d}.jpg')
                        cv2.imwrite(save_path, full_canvas)
                    except Exception as e:
                        pass


        out_flat = torch.stack(enhanced_seq, dim=1).view(bs_t, self.c1, h, w)

        return out_flat


class CoarseGuidanceHead(nn.Module):



    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 3, padding=1),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 4, 1, 1),
            nn.Sigmoid()
        )


        nn.init.constant_(self.conv[3].bias, -2.0)

    def forward(self, x):
        return self.conv(x)

class TemporalAttention(nn.Module):
    """Temporal self-attention over a fixed-length frame sequence."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, seq_len):
        bt, n, c = x.shape
        if seq_len <= 1 or bt % seq_len != 0:
            return x
        b = bt // seq_len
        x_time = x.view(b, seq_len, n, c).transpose(1, 2).reshape(b * n, seq_len, c)
        residual = x_time
        x_time = self.norm(x_time)
        x_attn, _ = self.attn(x_time, x_time, x_time)
        x_time = residual + x_attn
        return x_time.view(b, n, seq_len, c).transpose(1, 2).reshape(bt, n, c)


class DistanceMaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.dist_decay = nn.Parameter(torch.tensor(0.002))

    def forward(self, x, guidance_map):
        b, n, c = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.normalize(q.float(), dim=-1, p=2)
        k = F.normalize(k.float(), dim=-1, p=2)
        logit_scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale

        if guidance_map is not None:
            gh, gw = guidance_map.shape[-2:]
            if gh * gw != n:
                side = int(math.sqrt(n))
                gh = gw = side
            y_range = torch.arange(gh, device=x.device, dtype=torch.float32)
            x_range = torch.arange(gw, device=x.device, dtype=torch.float32)
            try:
                yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
            except TypeError:
                yy, xx = torch.meshgrid(y_range, x_range)
            coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
            if coords.shape[0] == n:
                dist = torch.cdist(coords, coords, p=2)
                attn = attn - torch.abs(self.dist_decay.float()) * dist.view(1, 1, n, n)

            mask = guidance_map.float()
            if mask.shape[-2] * mask.shape[-1] != n:
                mask = F.interpolate(mask, size=(gh, gw), mode='bilinear', align_corners=False)
            mask_flat = mask.flatten(2)
            if mask_flat.shape[-1] == n:
                penalty = (1.0 - mask_flat.unsqueeze(1)) * -100.0
                attn = attn + penalty

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
        if self.training:
            attn.register_hook(lambda grad: torch.clamp(grad, -10.0, 10.0))
        x_out = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, c)
        x_out = self.proj(x_out)
        return self.proj_drop(x_out)


class AutoregressiveGuidedLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(c)
        self.self_attn = DistanceMaskedAttention(c, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, 4 * c),
            nn.SiLU(),
            nn.Linear(4 * c, c),
            nn.Dropout(0.1),
        )
        self.norm3 = nn.LayerNorm(c)

    def forward(self, curr_feat, track_memory, guidance_map):
        corrected_memory, _ = self.cross_attn(
            query=self.norm1(track_memory),
            key=self.norm1(curr_feat),
            value=self.norm1(curr_feat),
        )
        track_memory = track_memory + corrected_memory
        track_memory = track_memory + self.self_attn(self.norm2(track_memory), guidance_map)
        track_memory = track_memory + self.ffn(self.norm3(track_memory))
        return track_memory

class GlobalClusterBlock(nn.Module):
    def __init__(self, c1, c2, heads=4, num_layers=1, seq_len=5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.seq_len = seq_len
        self.shortcut = nn.Conv2d(c1, c2, 1)
        self.coarse_head = CoarseGuidanceHead(c1)
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.pos_shape = (64, 64)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pos_shape[0] * self.pos_shape[1], c2))
        self.init_track_query = nn.Parameter(torch.zeros(1, self.pos_shape[0] * self.pos_shape[1], c2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.init_track_query, std=0.02)
        self.layers = nn.ModuleList([AutoregressiveGuidedLayer(c2, heads) for _ in range(num_layers)])
        self.step_counter = 0

    def resize_pos_embed(self, pos_embed, shape):
        h, w = shape
        if h * w == pos_embed.shape[1]:
            return pos_embed
        c = pos_embed.shape[-1]
        pos_embed = pos_embed.transpose(1, 2).reshape(1, c, self.pos_shape[0], self.pos_shape[1])
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        return pos_embed.flatten(2).transpose(1, 2)

    def forward(self, x):
        bt, _, h, w = x.shape
        guidance_map = self.coarse_head(x)
        guidance_map = torch.nan_to_num(guidance_map, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        feat_shortcut = self.shortcut(x)
        feat = self.proj(x)
        c = feat.shape[1]
        flatten_x = feat.flatten(2).transpose(1, 2)
        flatten_x = flatten_x + self.resize_pos_embed(self.pos_embed, (h, w))

        if self.training and bt % self.seq_len == 0 and bt >= self.seq_len:
            b = bt // self.seq_len
            flatten_seq = flatten_x.view(b, self.seq_len, h * w, c)
            guidance_seq = guidance_map.view(b, self.seq_len, 1, h, w)
            track_memory = self.resize_pos_embed(self.init_track_query, (h, w)).repeat(b, 1, 1)
            enhanced_seq = []
            for t in range(self.seq_len):
                curr_feat = flatten_seq[:, t]
                curr_mask = guidance_seq[:, t]
                for layer in self.layers:
                    track_memory = layer(curr_feat, track_memory, curr_mask)
                enhanced_seq.append(curr_feat + track_memory)
            feat_refined = torch.stack(enhanced_seq, dim=1).view(bt, h * w, c)
        else:
            track_memory = self.resize_pos_embed(self.init_track_query, (h, w)).repeat(bt, 1, 1)
            for layer in self.layers:
                track_memory = layer(flatten_x, track_memory, guidance_map)
            feat_refined = flatten_x + track_memory

        feat_refined = feat_refined.transpose(1, 2).reshape(bt, c, h, w)
        feat_final = feat_shortcut + feat_refined

        if self.training:
            global GLOBAL_MASK_DICT
            GLOBAL_MASK_DICT[x.device] = guidance_map

        return feat_final


class C3_ASSCA(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, seq_len=5, gamma_scale=0.1):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.assca = ASSCAModule(c2, seq_len=seq_len, gamma_scale=gamma_scale)

    def forward(self, x):
        c3_out = super().forward(x)
        return self.assca(c3_out)

class SPPF_Transformer(SPPF):
    def __init__(self, c1, c2, k=5, seq_len=5, num_heads=8):
        super().__init__(c1, c2, k)
        self.transformer = GlobalClusterBlock(c2, c2, num_heads, 1)

    def forward(self, x):
        sppf_out = super().forward(x)
        return self.transformer(sppf_out)


