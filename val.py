
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data data/infrared_e2.yaml --weights weights/best.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
from numpy import seterr
seterr('raise')
import torch
from tqdm import tqdm
import csv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend, GLOBAL_ORACLE_PERIPHERAL_DICT
from models.experimental import attempt_load
from LSTM_video_loader import VideoFourierDataset
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size,
                           coco80_to_coco91_class, colorstr, increment_path, print_args,
                           scale_coords,scale_coords_poly, xywh2xyxy, xyxy2xywh,
                           get_ft_num)
from utils.post_process import non_max_suppression, non_max_suppression_ft
from utils.ft_iou import box_iou_ft

from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.loss import loop_loss
from detect import detect, detect_with_cms
from general.MyString import replace_last_path

from DOTA_devkit.polyiou_cpu import poly_iou_cpu64
import pickle

def save_one_txt(predn, save_conf, shape, file):

    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches

def vector_dot(labels1, labels2):
    """
    labels1: (N, 8)
    labels2: (N, 8)
    """
    device = labels1.device
    n = labels1.shape[0]
    m = labels2.shape[0]
    dot = torch.zeros((n, m), dtype=torch.float32).to(device)
    if n:
        vec1 = rbox2vec(labels1)
        vec2 = rbox2vec(labels2)
        s1 = torch.sqrt((vec1 ** 2).sum(dim=1))
        s2 = torch.sqrt((vec2 ** 2).sum(dim=1))
        for i in range(n):
            for j in range(m):
                cos_t = (vec1[i] * vec2[j]).sum() / s1[i] * s2[j]
                dot[i][j] = cos_t
    return dot


def rbox2vec(rbox):
    cx = rbox[:, ::2].sum(dim=1) / 4
    cy = rbox[:, 1::2].sum(dim=1) / 4
    px = (rbox[:, 0] + rbox[:, 2]) / 2
    py = (rbox[:, 1] + rbox[:, 3]) / 2
    dx = px-cx
    dy = py-cy
    L = torch.sqrt(dx ** 2 + dy ** 2)
    return torch.stack((dx/L, dy/L), dim=1)


def process_batch_poly(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2, x3,y3, x4,y4) format.
    Arguments:
        detections (Array[N, 10]), x1, y1, x2, y2, x3,y3, x4,y4 conf, class
        labels (Array[M, 9]), class, x1, y1, x2, y2, x3,y3, x4,y4
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    iou = poly_iou_cpu64(labels[:, 1:].cpu().numpy().astype(np.float64), detections[:, :8].cpu().numpy().astype(np.float64))
    iou = torch.from_numpy(iou).to(labels.device)


    dot = vector_dot(labels[:, 1:], detections[:, :8])


    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 9]))

    assert(len(x)==2 and x[0].shape[0]==x[1].shape[0])
    if x[0].shape[0]:


        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()

        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        matches = torch.Tensor(matches).to(iouv.device)

        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv

    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches

from utils.ft_iou import polygon_iou_ft

def process_batch_ft(detections_ft, labels_ft, d_c, l_c, iouv, polygon=True):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections_ft (Array[N, 2+4*coef]), x1, y1, x2, y2, conf, class
        labels_ft (Array[M, 2+4*coef]), class, x1, y1, x2, y2
        d_c, l_c
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections_ft.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)

    if polygon:
        iou = torch.from_numpy(polygon_iou_ft(labels_ft, detections_ft, l_c, d_c)).to(iouv.device).float()
    else:
        iou = box_iou_ft(labels_ft, detections_ft)
    x = torch.where((iou >= iouv[0]) & (l_c[:, 0:1] == d_c))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0,3]).to(iouv.device)
    return correct, matches


    """
     class_match  IoU
    """


    class_match = (l_c.view(-1, 1) == d_c.view(1, -1))
    target_shape = class_match.shape


    if polygon:
        iou = torch.from_numpy(polygon_iou_ft(labels_ft, detections_ft, l_c, d_c)).to(iouv.device).float()
    else:
        iou = box_iou_ft(labels_ft, detections_ft)


    if iou.shape != target_shape:
        print(f"\n[DEBUG] Shape Conflict Detected!")
        print(f"  > Target (ClassMatch): {target_shape}")
        print(f"  > Current (IoU):       {iou.shape}")


    if iou.shape != target_shape:

        if iou.t().shape == target_shape:
            print("  > Action: Transposing IoU to match target.")
            iou = iou.t()
        else:

            print("  > CRITICAL ERROR: Transpose failed to match dimensions!")


            return torch.zeros(detections_ft.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device), torch.zeros([0, 3]).to(iouv.device)


    correct = torch.zeros(detections_ft.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)


    x = torch.where((iou >= iouv[0]) & class_match)

    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    else:
        matches = torch.zeros([0, 3]).to(iouv.device)

    return correct, matches

@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        imgsz=[640,640],
        conf_thres=0.001,
        ab_thres=3.0,
        iou_thres=0.45,
        task='val',
        device='',
        single_cls=False,
        augment=False,
        verbose=False,
        save_txt=False,
        save_conf=False,
        project=ROOT / 'runs/val',
        name='exp',
        exist_ok=False,
        half=True,
        dnn=False,
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        fold=2,
        pts=None,
        nms_polygon = 1,
        map_ft = True,
        polygon=False,
        legacy_t2_metrics=False,
        workers=2,
        ):

    training = model is not None

    if training:
        device, pt = next(model.parameters()).device, True

        half &= device.type != 'cpu'
        model.half() if half else model.float()
        pts = model.get_module_byname('Detect2') is not None if pts is None else pts
        ftinfer_m = model.get_module_byname('FTInfer')
        if ftinfer_m is not None:
            ft_length = ftinfer_m.ft_length
            ft_infer = True
        else:
            ft_length = 0
            ft_infer = False
    else:
        device = select_device(device, batch_size=batch_size)


        if project!='' and project is not None:
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        else:
            weights_path = os.path.dirname(weights)
            train_path = os.path.dirname(weights_path) if os.path.basename(weights_path)=='weights' else weights_path
            assert os.path.isdir(train_path)
            save_dir = increment_path(Path(train_path) / 'val', exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)


        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, pt = model.stride, model.pt
        pts = model.model.get_module_byname('Detect2') is not None
        ft_infer = False

        ft_length = 0
        p = model.model.get_module_byname('FTInfer')
        if p:
            ft_infer = True
            ft_length = p.ft_length
        imgsz = check_img_size(imgsz, s=stride)
        half &= pt and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
        else:
            half = False
            batch_size = 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')


        data = check_dataset(data)

        train_path = data['train']
        ft_coef = data.get('ft_coef', 0)
        if ft_coef == -1:
            ft_coef = get_ft_num(train_path)


    model.eval()
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()


    if not training:
        imgsz2 = [imgsz, imgsz] if isinstance(imgsz, int) else imgsz
        model_cfg = getattr(model.model, 'yaml', {})
        model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
        point_temporal = bool(model_cfg.get('point_temporal_enabled', False))
        fourier_temporal = bool(model_cfg.get('ft_temporal_enabled', False))
        temporal_seq_len = max(
            int(model_cfg.get('point_temporal_seq_len', 1)) if point_temporal else 1,
            int(model_cfg.get('ft_temporal_seq_len', 1)) if fourier_temporal else 1)


        if pt and device.type != 'cpu' and temporal_seq_len == 1:
            model(torch.zeros(1, getattr(model.model, 'ch', 3), imgsz2[0], imgsz2[1]).to(device).type_as(next(model.model.parameters())))
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'


        hyp = model.model.hyp


        if temporal_seq_len > 1:
            LOGGER.info(
                f'Temporal validation loader: VideoFourierDataset, '
                f'T={temporal_seq_len}, final frame per sliding clip')
            dataset = VideoFourierDataset(
                path=data[task], img_size=imgsz2,
                seq_length=temporal_seq_len, ft_coef=ft_coef,
                augment=False, hyp=hyp)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True,
                collate_fn=VideoFourierDataset.collate_fn)
        else:

            val_count = data.get('val_count', 0)
            dataloader, dataset = create_dataloader(
                data[task], imgsz2, batch_size, stride, single_cls,
                rect=False, pad=pad, hyp=hyp,
                prefix=colorstr(f'{task}: '), pts=pts,
                ft_coef=ft_coef, workers=workers,
                debug_samples=0, sample_count=val_count)

        if getattr(model.model, 'ch', 3) != 3:
            conv1 = model.model.model[0].conv.weight.detach().clone().permute(1,0,2,3).reshape(model.model.ch, -1)
            conv1 = torch.abs(conv1).sum(-1)
            conv1_wts = torch.softmax(conv1, 0)
            conv1_idx = torch.argsort(conv1_wts, descending=True)
            print(f'\033[32m{conv1_idx}\033[0m')
            print(f'\033[32m{conv1_wts[conv1_idx]}\033[0m')

    mask_dir = data.get('mask_dir', 0)
    assert isinstance(mask_dir, (int, list)), f'{mask_dir}, Mask Dir Error.'
    mask_dir = [mask_dir] * nc if isinstance(mask_dir, int) else mask_dir
    if len(mask_dir) != nc:
        mask_dir = mask_dir + [0] * (nc - len(mask_dir))

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}


    metric_model = model.module if hasattr(model, 'module') else model
    model_yaml = getattr(metric_model, 'yaml', {})
    if not isinstance(model_yaml, dict) or not model_yaml:


        wrapped_model = getattr(metric_model, 'model', None)
        model_yaml = getattr(wrapped_model, 'yaml', {})
    formation_macro_detect = bool(model_yaml.get('formation_macro_detect', False)) if isinstance(model_yaml, dict) else False
    oracle_switch = os.getenv('ORACLE_GT_ENABLED', '1').strip().lower()
    oracle_peripheral_gt = (
        bool(model_yaml.get('oracle_peripheral_gt', False)) and
        oracle_switch not in {'0', 'false', 'no', 'off'}
    ) if isinstance(model_yaml, dict) else False


    if polygon:
        s = ('%15s' + '%12s' * 10) % (
            'Class', 'Images', 'MacroGT', 'Poly-P', 'Poly-R', 'Poly-mAP50',
            'Poly-mAP', 'Poly-F1', 'conf*', 'ctr_px', 'harm_px')
    else:
        s = ('%15s' + '%12s' * 10) % (
            'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5',
            'mAP@.5:.95', 'f1', 'thresh', 'ft_xy', 'ft_coef')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(5, device=device)
    stats, ap, ap_class = [], [], []

    ft_center_val = torch.zeros(1, device=device)
    ft_coef = max((ft_length - 2) // 4, 0)
    ft_coef_val = torch.zeros(ft_coef, device=device)
    ft_num = torch.zeros(1, device=device)
    if ft_infer and map_ft:
        metric_shape = ('complete reconstructed K-order polygon'
                        if polygon else 'legacy first-order surrogate polygon')
        LOGGER.info(
            f'Fourier validation metric: {metric_shape}; '
            'P/R/F1 use the maximum-F1 confidence point; AP uses conf>=%.4g.'
            % conf_thres)
    for batch_i, (im, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        seq_len = int(getattr(getattr(dataloader, 'dataset', None), 'seq_len', 1))
        if im.ndimension() == 5:
            seq_len = int(im.shape[1])


        targets = targets.to(device)
        targets_all = targets.clone()
        if formation_macro_detect:
            targets = targets[targets[:, 1] == 1.0].clone()
            if targets.shape[0] > 0:
                targets[:, 1] = 0.0
        else:
            targets = targets[targets[:, 1] == 0.0]


        if seq_len > 1 and targets.shape[0] > 0:
            last_target = targets[:, 0].long() % seq_len == (seq_len - 1)
            targets = targets[last_target].clone()
            targets[:, 0] = torch.div(targets[:, 0], seq_len, rounding_mode='floor')


        targets6 = targets[..., :6].clone().to(device)
        if pt:
            im = im.to(device, non_blocking=True)
        im = im.half() if half else im.float()


        if im.ndimension() == 5:
            b, t, c, h, w = im.shape
            seq_len = t
            im = im.view(b * t, c, h, w)


        nb, _, height, width = im.shape

        im /= 255
        nb, _, height, width = im.shape
        t2 = time_sync()
        dt[0] += t2 - t1


        if not pts:
            if oracle_peripheral_gt:
                GLOBAL_ORACLE_PERIPHERAL_DICT[im.device] = {
                    'targets': targets_all.to(im.device).detach()
                }
            else:
                GLOBAL_ORACLE_PERIPHERAL_DICT.pop(im.device, None)
            if training:
                out,train_out = [],[]
                tmp = model(im, augment=augment)
                if isinstance(tmp, list) and len(tmp) > 2:
                    tmp = tmp[:2]
                for t in tmp:
                    out.append(t[0])
                    train_out.extend(t[1])
            else:
                out, train_out = model(im, augment=augment, val=True)
                if isinstance(out, list) and len(out) > 2:
                    out = out[:2]
                if isinstance(train_out, list) and len(train_out) > 2:
                    train_out = train_out[:2]
            dt[1] += time_sync() - t2


            if seq_len > 1:
                last_idx = torch.arange(seq_len - 1, im.shape[0], seq_len, device=device)
                def filter_last(p):
                    if isinstance(p, dict):
                        return {k: filter_last(v) for k, v in p.items()}
                    if isinstance(p, (list, tuple)): return type(p)(filter_last(x) for x in p)
                    elif isinstance(p, torch.Tensor): return p[last_idx]
                    return p
                train_out = filter_last(train_out)


            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets_all.to(device), fold=fold,
                                     mask_dir=mask_dir, paths=paths)[1]
                n_loop = compute_loss.n_loop
            else:
                n_loop = 8

            targets6[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = []

            pbox_out = []
            for i in out[0]:
                pbox_out.append(i.view(i.shape[0], -1, i.shape[-1]))
            pCms = [None] * len(out[0])
            pFt = [None] * len(out[0])
            sv_id = 0
            if ft_infer:
                sv_id = 1
                pFt = out[sv_id]
                for i, j in enumerate(pFt):
                    pFt[i] = j.view(j.shape[0], -1, j.shape[-1])
                pFt = torch.cat(pFt, 1)
            pFt = None if isinstance(pFt, list) else pFt

            pbox_out = torch.cat(pbox_out, 1)

            t3 = time_sync()

            if not (nms_polygon and ft_infer):
                out, indices = non_max_suppression(pbox_out, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls, return_indices=True)

            else:


                out, out_ft, indices = non_max_suppression_ft(
                    pbox_out, pFt, conf_thres, iou_thres, labels=lb,
                    multi_label=False, agnostic=single_cls,
                    return_indices=True, polygon=nms_polygon == 2)


            if seq_len > 1:

                out = out[(seq_len - 1)::seq_len]


                if (nms_polygon and ft_infer):
                    out_ft = out_ft[(seq_len - 1)::seq_len]


                if 'indices' in locals():
                    indices = indices[(seq_len - 1)::seq_len]


                if 'pFt' in locals() and pFt is not None:
                    pFt = pFt[(seq_len - 1)::seq_len]

                paths = paths[(seq_len - 1)::seq_len]
                shapes = shapes[(seq_len - 1)::seq_len]


            dt[2] += time_sync() - t3


            for si, pred in enumerate(out):
                labels = targets6[targets6[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue


                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()


                if isinstance(shapes[si][0], int):


                    scale_coords(im[si].shape[1:], predn[:, :4], shapes[si], None)
                else:

                    scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])


                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])

                    if isinstance(shapes[si][0], int):


                        scale_coords(im[si].shape[1:], predn[:, :4], shapes[si], None)
                    else:

                        scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])


                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)

                    if ft_length > 0:


                        labels_ft = targets[targets[:, 0] == si, 6:6+ft_length].clone()
                        labels_ft[:, :2] = labels_ft[:, :2] * torch.Tensor([width, height]).to(device)
                        labels_ft[:, 2:] = labels_ft[:, 2:] * torch.Tensor([width, width, height, height]).to(device).repeat((ft_length-2) // 4).view(1, -1)


                    if (map_ft and ft_infer):
                        correct, matches = process_batch_ft(detections_ft=pFt[si][indices[si]],
                                                            labels_ft=labels_ft,
                                                            d_c=predn[:, 5],
                                                            l_c=labels[:, 0:1],
                                                            iouv=iouv,
                                                            polygon=polygon)

                    else:
                        correct, matches = process_batch(predn, labelsn, iouv)
                    if matches.shape[0] > 0:
                        if ft_infer:
                            t_idx, p_idx, _ = torch.chunk(matches, 3, dim=-1)
                            t_idx, p_idx = t_idx.view(-1).long(), p_idx.view(-1).long()

                            if ft_infer:
                                pFt_i = pFt[si][indices[si]]
                                ft_labels_xy = labels_ft[:, :2].clone()


                                ft_labels_coef = labels_ft[:, 2:].clone()
                                assert t_idx.shape[0]<=min(len(pFt[si][indices[si]]),len(labels_ft))


                                center_error = torch.sqrt(
                                    ((pFt_i[p_idx][:, :2] - ft_labels_xy[t_idx]) ** 2).sum(-1)
                                )
                                if not legacy_t2_metrics:


                                    ft_center_val += center_error.sum()
                                else:


                                    ft_center_val += center_error.mean()
                                ft_coef_min = loop_loss(n_loop,ft_labels_coef[t_idx],pFt_i[p_idx][:, 2:])
                                temp1 = pFt_i[p_idx][:, 2:].clone().view(-1, ft_coef, 4).float()
                                temp2 = ft_coef_min.view(-1, ft_coef, 4).float()
                                ft_coef_val = ft_coef_val + torch.abs(torch.sqrt((temp1 * temp1).sum(-1)) - torch.sqrt((temp2 * temp2).sum(-1))).sum(0)

                                ft_num += t_idx.shape[0]
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
        else:
            targets[...,2:10] = targets[..., 6:14].clone()
            targets = targets[..., :10]


            out, train_out = detect_with_cms(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir, dt=dt)


            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets_all.to(device), fold=fold,
                                     mask_dir=mask_dir, paths=paths)[1]


            targets[:, 2:] *= torch.Tensor([width, height, width, height,width, height, width, height]).to(device)

            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                path, shape = Path(paths[si]), shapes[si][0]
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue


                predn = pred.clone()
                scale_coords_poly(im.shape[2:], predn[:, :8], shape, shapes[si][1])


                if nl:
                    tpts = labels[:, 1:]
                    scale_coords_poly(im[si].shape[1:], tpts, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tpts), 1)
                    correct, matches = process_batch_poly(predn[:, :10], labelsn, iouv)


                    if plots:
                        confusion_matrix.process_batch_poly(predn[:, :10], labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

                stats.append((correct.cpu(), pred[:, 8].cpu(), pred[:, 9].cpu(), tcls))


                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                callbacks.run('on_val_image_end', pred, predn, path, names, im[si])


    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names, cut=True)


        with open(save_dir / 'status.pkl', 'wb') as f:
            pickle.dump([py,ap,names], f)


        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
        train_path = weights.parent.parent if not training and not isinstance(weights, str) else save_dir
        np.save(train_path / 'threshs.npy', threshs)
    else:

        f1 = torch.zeros(1)
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
        threshs = torch.zeros(1)

    ft_center_val = ft_center_val / (ft_num.sum() + 1e-8)
    ft_coef_val = ft_coef_val / (ft_num + 1e-8)


    pf = '%15s' + '%12i' * 2 + '%12.4g' * 6
    LOGGER.info(('\033[32m' + pf + '%12.4g' * 2 + '\033[0m') % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(),
                threshs.mean(),
                ft_center_val, ft_coef_val.mean()))
    if (verbose or (not training)) and nc >= 1 and len(stats):
        if nc < 100:
            for i, c in enumerate(ap_class):
                LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]))
        with open(save_dir / 'classes_map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            if polygon:
                writer.writerow([
                    'Class', 'Images', 'MacroGT', 'Poly-P', 'Poly-R',
                    'Poly-mAP@.5', 'Poly-mAP@.5:.95', 'Poly-F1', 'conf*',
                    'center_error_px', 'harmonic_amplitude_error_px'])
            else:
                writer.writerow([
                    'Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5',
                    'mAP@.5:.95', 'f1', 'thresh', 'ft_xy', 'ft_coef'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [
                "all", seen, nt.sum(), mp, mr, map50, map, f1.mean(),
                threshs.mean(), ft_center_val.item(), ft_coef_val.mean().item()]])

            for i, c in enumerate(ap_class):
                class_name = names.get(c, "Unknown")
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [
                    class_name, seen, nt[c], p[i], r[i], ap50[i], ap[i],
                    f1[i], threshs[i], '', '']])


    t = tuple(x / seen * 1E3 for x in dt)
    if not training:
        imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz
        shape = (batch_size, 3, imgsz2[0], imgsz2[1])
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)


    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')


    model.float()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to \033[32m{colorstr('bold', save_dir)}\033[0m{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, f1.mean(), *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/infrared_e2.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640,640], help='inference size h w (pixels)')
    parser.add_argument('--conf-thres', '--conf_thres', dest='conf_thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', '--iou_thres', dest='iou_thres', type=float, default=0.20, help='NMS IoU threshold')
    parser.add_argument('--ab-thres', '--ab_thres', dest='ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--fold', type=int, default=2, help='')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt save_txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--nms-polygon', '--nms_polygon', dest='nms_polygon', type=int, default=1, help='2=full Fourier polygon NMS, 1=first-order polygon NMS, 0=HBB NMS')
    parser.add_argument('--map-ft', '--map_ft', dest='map_ft', action='store_true', default=True, help='use Fourier contours to calculate AP')
    parser.add_argument('--no-map-ft', dest='map_ft', action='store_false', help='calculate AP from boxes instead of Fourier contours')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--polygon', dest='polygon', action='store_true', default=False, help='match the complete reconstructed K-order Fourier curve')
    parser.add_argument('--no-polygon', dest='polygon', action='store_false', help='use the legacy first-order surrogate for matching (default)')
    parser.add_argument('--legacy-t2-metrics', action='store_true', help='use legacy Fourier metric accumulation')
    opt = parser.parse_args()


    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    if opt.task in ('train', 'val', 'test') and isinstance(opt.weights, list) and len(opt.weights) == 1:
        opt.weights = opt.weights[0]
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    if opt.task in ('train', 'val', 'test'):
        if opt.conf_thres > 0.001:
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        if 0:
            name = opt.name + '_' + Path(opt.data).stem
            name_s = ['hv', 'ft']
            iou_t = [0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
            for i in range(2):
                opt.nms_polygon = 1 if name_s[i] == 'ft' else 0
                for j in range(len(iou_t)):
                    opt.name = name + f'_{name_s[i]}{iou_t[j]*100:.0f}'
                    opt.iou_thres = iou_t[j]
                    run(**vars(opt))
        else:
            run(**vars(opt))


    elif opt.task == 'speed':

        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':

        x = list(range(256, 1536 + 128, 128))
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'
            y = []
            for i in x:
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)
            np.savetxt(f, y, fmt='%10.4g')
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
