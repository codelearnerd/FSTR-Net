
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhr2xyxyxyxy
from utils.metrics import bbox_ioa,bbox_ioas
from DOTA_devkit.polyiou_cpu import poly_iou_cpu64

from utils.ft_utils import ffts_1,ffts_1_slow
import torch

def augment_ft(ft_labels,A23):
    is_numpy = isinstance(ft_labels, (np.ndarray,))
    if is_numpy:
        n = ft_labels.shape[0]
        ft_xy = np.split(ft_labels[:, :2], 2, axis=-1)

        ft_coef = np.split(ft_labels[:, 2:].reshape(ft_labels.shape[0], -1, 4), 4, axis=-1)
        x_coef = np.concatenate([ft_xy[0], ft_coef[0].squeeze(-1), ft_coef[1].squeeze(-1)], dtype=np.float32, axis=-1)
        y_coef = np.concatenate([ft_xy[1], ft_coef[2].squeeze(-1), ft_coef[3].squeeze(-1)], dtype=np.float32, axis=-1)
        ft = np.zeros((n, x_coef.shape[-1], 3))
        ft[:, :, 0] = x_coef
        ft[:, :, 1] = y_coef
        ft[:, 0, 2] = 1
        ft = ft.reshape([-1, 3])
        ft = (ft @ A23.T).reshape(n, x_coef.shape[-1], 2)
        ft_xy = ft[:, 0]
        ft_coef = [ft_t.squeeze(-1) for ft_t in np.split(ft[:, 1:], 2, axis=-1)]
        ft_ab, ft_cd = [[ft_t[..., None] for ft_t in np.split(ft_c, 2, axis=-1)] for ft_c in ft_coef]
        ft_coef = np.concatenate([ft_ab[0], ft_ab[1], ft_cd[0], ft_cd[1]], axis=-1).reshape(n, -1)
        ft_coef = np.concatenate([ft_xy, ft_coef], axis=-1)
        assert ft_coef.shape==ft_labels.shape
        return ft_coef
    else:
        n = ft_labels.shape[0]
        device, dtype = ft_labels.device, ft_labels.dtype


        A23 = A23.to(device=device, dtype=dtype)


        ft_xy_x, ft_xy_y = torch.split(ft_labels[:, :2], 1, dim=-1)


        coef_blocks = ft_labels[:, 2:].reshape(n, -1, 4)
        a, b, c, d = [t.squeeze(-1) for t in torch.split(coef_blocks, 1, dim=-1)]


        x_coef = torch.cat([ft_xy_x, a, b], dim=-1)
        y_coef = torch.cat([ft_xy_y, c, d], dim=-1)


        ft = torch.zeros((n, x_coef.shape[-1], 3), dtype=dtype, device=device)
        ft[:, :, 0] = x_coef
        ft[:, :, 1] = y_coef
        ft[:, 0, 2] = 1


        ft = (ft.reshape(-1, 3) @ A23.t()).reshape(n, x_coef.shape[-1], -1)
        if ft.shape[-1] != 2:
            ft = ft[..., :2]


        ft_xy = ft[:, 0, :]
        abcd_split = torch.split(ft[:, 1:, :], 1, dim=-1)
        ab = torch.chunk(abcd_split[0].squeeze(-1), 2, dim=-1)
        cd = torch.chunk(abcd_split[1].squeeze(-1), 2, dim=-1)


        coef_restored = torch.cat(
            [ab[0].unsqueeze(-1), ab[1].unsqueeze(-1), cd[0].unsqueeze(-1), cd[1].unsqueeze(-1)],
            dim=-1
        ).reshape(n, -1)


        ft_coef = torch.cat([ft_xy, coef_restored], dim=-1)

        assert ft_coef.shape==ft_labels.shape
        return ft_coef

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):

    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)


def hist_equalize(im, clahe=True, bgr=False):

    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)


def replicate(im, labels):

    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2
    for i in s.argsort()[:round(s.size * 0.5)]:
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):

    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)


    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if im.shape[2] <= 4:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        im = _numpy_pad_multichannel(im, top, bottom, left, right, color)
    return im, ratio, (dw, dh)


def random_perspective_old(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0), ft_length=0):


    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2


    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2


    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)


    R = np.eye(3)
    a = random.uniform(-degrees, degrees)

    s = random.uniform(1 - scale, 1 + scale)

    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)


    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)


    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height


    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))


    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]


                new[i] = segment2box(xy, width, height)

        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)


            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T


        if ft_length > 0:
            ft_labels = targets[:,5:5+ft_length]
            ft_xy = np.split(ft_labels[:, :2], 2, axis=-1)
            ft_coef = np.split(ft_labels[:, 2:].reshape(ft_labels.shape[0], -1, 4), 4, axis=-1)
            x_coef = np.concatenate([ft_xy[0], ft_coef[0].squeeze(-1), ft_coef[1].squeeze(-1)], dtype=np.float32, axis=-1)
            y_coef = np.concatenate([ft_xy[1], ft_coef[2].squeeze(-1), ft_coef[3].squeeze(-1)], dtype=np.float32, axis=-1)
            ft = np.zeros((n, ft_length // 2, 3))
            ft[:, :, 0] = x_coef
            ft[:, :, 1] = y_coef
            ft[:, 0, 2] = 1
            ft = ft.reshape([-1, 3])
            ft = (ft @ M.T)[:, :2].reshape(n, ft_length // 2, 2)
            ft_xy = ft[:, 0]
            ft_coef = [ft_t.squeeze(-1) for ft_t in np.split(ft[:, 1:], 2, axis=-1)]
            ft_ab, ft_cd = [[ft_t[..., None] for ft_t in np.split(ft_c, 2, axis=-1)] for ft_c in ft_coef]
            ft_coef = np.concatenate([ft_ab[0], ft_ab[1], ft_cd[0], ft_cd[1]], axis=-1).reshape(n, -1)
            ft_coef = np.concatenate([ft_xy, ft_coef], axis=-1)


        i = box_candidates(boxes=new, width=width,height=height)

        targets = targets[i]
        targets[:, 1:5] = new[i]
        if ft_length > 0:
            targets[:, 5:5 + ft_length] = ft_coef[i]

    return im, targets

def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, flip = [0.5,0.5],
                       border=(0, 0), ft_length=0, pts=False, iou_thr = 0.3, auto_aug_pts=1, clip_rate=0.2, clip_robx=False):


    sindex = 13 if pts else 5
    assert ((ft_length - 2) % 4) == 0
    nf = (ft_length - 2) // 4

    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2


    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2


    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)


    R = np.eye(3)
    a = random.uniform(-degrees, degrees)

    s = random.uniform(1 - scale, 1 + scale)

    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)


    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)


    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height


    Fud = np.eye(3)
    if random.uniform(0,1) < flip[1]:
        Fud[1, 1] = -1
        Fud[1, 2] = height
    Flr = np.eye(3)
    if random.uniform(0,1) < flip[0]:
        Flr[0, 0] = -1
        Flr[0, 2] = width


    M = T @ S @ R @ P @ C @ Fud @ Flr
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))


    n = len(targets)
    if n:
        ft_labels = targets[:,sindex: sindex + ft_length]
        ft_coef = augment_ft(ft_labels,M[:2])


        if auto_aug_pts==0:
            npts = 4
            xy = np.ones((n * npts, 3))
            if not pts:
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * npts, 2)
            else:
                xy[:, :2] = targets[:, 5:13].reshape(n * npts, 2)

            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, npts*2)


            i,ioa = out_range_filt_new(xy, im.shape[:2], iou_thr)
        else:
            npts = 64
            xy = np.ones((n * npts, 3))

            xf,yf = ffts_1(ft_coef,npts)
            xy = np.stack([xf, yf], axis=-1).reshape(-1, 2*npts)


            from .polygon import out_range_filt_poly_batch
            i,ioa = out_range_filt_poly_batch(xy, im.shape[:2], iou_thr)

        if i.shape[0] > 0:
            xy,ioa = xy[i],ioa[i]
            if clip_robx:
                xy = clip_xywhr_rboxes(xy, 0, 0, width, height, clip_rate)


            xy = xy.reshape(i.shape[0],npts,2)
            x = xy[:, :, 0]
            y = xy[:, :, 1]
            hbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, -1).T


            targets = targets[i]
            ft_coef = ft_coef[i]
            n = targets.shape[0]
            assert xy.shape[0]==targets.shape[0]
        else:
            targets = np.zeros((0, sindex + ft_length),dtype=np.float32)
            ft_coef = np.zeros((0, ft_length),dtype=np.float32)
            hbox = np.zeros((0,4),dtype=np.float32)
            n = 0

        assert hbox.shape[0]==targets.shape[0]
        if n>0:
            targets[:, 1:5] = hbox
            if pts:
                targets[:, 5:13] = xy
            if ft_length > 0:
                targets[:, sindex:sindex + ft_length] = ft_coef

    return im, targets

def cal_poly_area(pts8):
    points = pts8.reshape(4,2)

    points = np.vstack((points, points[0]))

    clockwise_sum = np.sum(points[:-1, 0] * points[1:, 1])
    counterclockwise_sum = np.sum(points[:-1, 1] * points[1:, 0])

    area = abs(clockwise_sum - counterclockwise_sum) / 2
    return area


def cal_poly_area_array(ptses):
    points = ptses.reshape(-1, 4,2)

    points = np.concatenate([points, points[:, 0:1, :]], axis=-2)

    clockwise_sum = np.sum(points[:, :-1, 0] * points[:, 1:, 1], axis=-1)
    counterclockwise_sum = np.sum(points[:, :-1, 1] * points[:, 1:, 0], axis=-1)

    area = abs(clockwise_sum - counterclockwise_sum) / 2
    return area

def out_range_filt_new(poly_ptses, shape, iou_thresh=0.2,ioa4=None):
    assert ioa4 is None or ioa4.shape[0]==poly_ptses.shape[0]
    height,width = shape
    image_poly = np.array([0, 0, width, 0, width,height, 0,height], dtype=np.float32).reshape(1, -1)
    image_area = width * height


    iou = poly_iou_cpu64(poly_ptses.astype(np.float64), image_poly.astype(np.float64)).reshape(-1)
    obj_area = cal_poly_area_array(poly_ptses)

    sec_area = iou * (image_area + obj_area) / (1 + iou)
    tmp = obj_area <= 0
    sec_area[tmp] = 0
    obj_area[tmp] = 1
    ioa = sec_area / obj_area
    if ioa4 is not None:
        assert ioa4.shape[0]==ioa.shape[0]
        ioa = np.minimum(ioa,ioa4)
    polys_out_ids = np.where(ioa > iou_thresh)[0]
    return polys_out_ids,ioa


def copy_paste(im, labels, segments, p=0.5):

    n = len(segments)
    if p and n:
        h, w, c = im.shape
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])
            if (ioa < 0.30).all():
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)
        i = result > 0

        im[i] = result[i]

    return im, labels, segments


def cutout(im, labels, p=0.5):

    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))


            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)


            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]


            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])
                labels = labels[ioa < 0.60]

    return labels


def mixup(im, labels, im2, labels2):

    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels

def box_candidates_old(box1, box2, width, height, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):

    cx,cy = (box1[2] + box1[0])/2, (box1[3] + box1[1])/2

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height) & (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)

def box_candidates(boxes, width, height):
    cx,cy = (boxes[:,2] + boxes[:,0])/2, (boxes[:,3] + boxes[:,1])/2
    return (cx>=0) & (cx<=width) & (cy>=0) & (cy<=height)


def box_candidates_ioa(boxes, width, height, iou_thr=0.2, w_min=0, h_min=0):
    image_box = np.array([w_min,h_min, width,height], dtype=np.float64)
    iou = bbox_ioas(image_box,boxes)
    assert np.all((iou >= 0) & (iou <= 1.0001)), "iou  [0, 1] "
    return iou>iou_thr

def resample_segments_v11(segments, n=100):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )
    return segments

def clip_xywhr_rboxes(xy, x1, y1, x2, y2, clip_rate=0.0):
    margin = int(clip_rate * min(x2-x1,y2-y1))
    assert margin>=0
    assert x1<x2 and y1<y2
    segments = [x for x in xy.copy().reshape(-1, 4, 2)]
    segments = resample_segments_v11(segments)
    bboxes = []
    for segment in segments:


        segment = segment.reshape(-1, 2)

        x, y = segment.T
        inside = (x >= x1-margin) & (y >= y1-margin) & (x <= x2+margin) & (y <= y2+margin)
        if inside.sum() > 0:
            points_inside = segment[inside]
            x1a, x2a, y1a, y2a = points_inside[:, 0].min(), points_inside[:, 0].max(), points_inside[:, 1].min(), points_inside[:, 1].max()

            segment[:, 0] = segment[:, 0].clip(x1a, x2a)
            segment[:, 1] = segment[:, 1].clip(y1a, y2a)

            (cx, cy), (w, h), angle = cv2.minAreaRect(segment)
            bboxes.append([cx, cy, w, h, angle / 180 * np.pi] if (w * h) > 0 else [0, 0, 0, 0, 0])
        else:
            bboxes.append([0, 0, 0, 0, 0])
    xy = xywhr2xyxyxyxy(np.array(bboxes, dtype=np.float32)).reshape(-1, 8)
    return xy

def _numpy_pad_multichannel(im, top, bottom, left, right, color):

    if isinstance(color, (int, float)):
        fill_value = color
    elif isinstance(color, (tuple, list)):
        fill_value = color[0]
    else:
        fill_value = 114
    if im.ndim == 3:
        num_channels = im.shape[2]
        pad_width = ((top, bottom), (left, right), (0, 0))
        constant_values = ((fill_value, fill_value), (fill_value, fill_value), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
        constant_values = fill_value
    im_padded = np.pad(im, pad_width, mode='constant', constant_values=constant_values)
    return im_padded
