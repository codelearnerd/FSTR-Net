
import os
import time
from subprocess import check_output

import numpy as np
import torch
import torchvision

from utils.metrics import box_iou

from utils.general import xywh2xyxy
from DOTA_devkit.polyiou_cpu import poly_nms_cpu64
from utils.ft_iou import polygon_iou_ft, polygon_nms_ft
from utils.geometry_safety import record_invalid_geometry
from utils.general import two_fold_angle,normalize_dir,ab2pts,box2poly
from utils.ft_utils import ft2pts

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    prediction: [batch, all_grid, 5+nc]
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    nc = prediction.shape[2] - 5
    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4] > conf_thres.min()


    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'


    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):


        x = x[xc[xi]]
        grid = indices_grid[xc[xi]]


        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)


        if not x.shape[0]:
            continue


        x[:, 5:] *= x[:, 4:5]


        box = xywh2xyxy(x[:, :4])


        if multi_label:
            i, j = (x[:, 5:] > conf_thres[None]).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            grid = grid[i]
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            filter_conf = (conf > conf_thres[j]).view(-1)
            x = torch.cat((box, conf, j.float()), 1)[filter_conf]
            grid = grid[filter_conf]


        if classes is not None:
            grid = grid[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]


        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):

            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output_indices[xi] = grid[i]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output, output_indices if return_indices else output

def non_max_suppression_ft(prediction, prediction_ft, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False, polygon=True):


    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()


    nc = prediction.shape[2] - 5
    if isinstance(conf_thres, float):
        conf_thres = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        conf_thres = conf_thres.to(prediction.device)
    xc = prediction[..., 4] > conf_thres.min()


    assert ((0 <= conf_thres) & (conf_thres <= 1)).all(), f'Invalid Confidence threshold [{conf_thres.min()}, {conf_thres.max()}], valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'


    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_ft = [torch.zeros((0, prediction_ft.shape[-1]), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 1), device=prediction.device)] * prediction.shape[0]

    for xi, (x,x_ft) in enumerate(zip(prediction, prediction_ft)):


        x = x[xc[xi]]
        x_ft = x_ft[xc[xi]]
        grid = indices_grid[xc[xi]]


        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)


        if x.shape[0] > 0:

            x[:, 5:] *= x[:, 4:5]


            box = xywh2xyxy(x[:, :4])


            if multi_label:
                i, j = (x[:, 5:] > conf_thres[None]).nonzero(as_tuple=False).T

                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float()), 1)
                x_ft = x_ft[i]
                grid = grid[i]
            else:

                conf, j = x[:, 5:].max(1, keepdim=True)


                obj_filt = (conf > conf_thres[j]).view(-1)
                x = torch.cat((box, conf, j.float()), 1)[obj_filt]

                x_ft = x_ft[obj_filt]
                grid = grid[obj_filt]


            if classes is not None:
                clsid = x[:, 5:6]
                grid = grid[(clsid == torch.tensor(classes, device=x.device)).any(1)]
                x = x[(clsid == torch.tensor(classes, device=x.device)).any(1)]
                x_ft = x_ft[(clsid == torch.tensor(classes, device=x.device)).any(1)]


            finite = torch.isfinite(x).all(1) & torch.isfinite(x_ft).all(1)
            if not bool(finite.all()):
                record_invalid_geometry('nonfinite_nms_prediction', int((~finite).sum()))
                x, x_ft, grid = x[finite], x_ft[finite], grid[finite]
            if not x.shape[0]:
                continue


            n = x.shape[0]
            if n > max_nms:
                keep_order = x[:, 4].argsort(descending=True)[:max_nms]
                grid, x, x_ft = grid[keep_order], x[keep_order], x_ft[keep_order]


            c = x[:, 5:6] * (0 if agnostic else max_wh)


            scores = x[:, 4]
            if x_ft.shape[0]>0:
                if polygon:
                    i = polygon_nms_ft(x_ft, scores, c[:,0].cpu().numpy(), iou_thres, cls_offset=0, dif_cls_iou_thresh=1.1)
                else:

                    polys = (ft2pts(x_ft) + c).cpu().numpy().astype(np.float64)
                    polys2 = np.concatenate([polys, scores.cpu().numpy().astype(np.float64).reshape(-1, 1)], axis=-1)
                    i = poly_nms_cpu64(polys2, iou_thres)
            else:
                i = np.array([], dtype=int)

            if i.shape[0] > max_det:
                i = i[:max_det]
            output_indices[xi] = grid[i]
            output[xi] = x[i]
            output_ft[xi] = x_ft[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break

    return (output, output_ft, output_indices) if return_indices else (output, output_ft)

def rot_nms(prediction, conf_thres=0.25, iou_thres=0.3, ab_thres=3.0, fold_angle=2, mask_dir=[], threshs=torch.zeros(0)):
    """Runs Non-Maximum Suppression (NMS) on inference results
    """

    pYolo = prediction[0]
    pRot = prediction[1]
    nl = len(pYolo)
    batchs = pYolo[0].shape[0]
    nc = pYolo[0].shape[-1] - 5


    targets = []
    tinfo = []
    for i in range(nl):

        yolo_map = pYolo[i]
        for batch_idx in range(batchs):
            batch_data = yolo_map[batch_idx]

            na = batch_data.shape[0]
            for anchor_idx in range(na):
                anchor_data = batch_data[anchor_idx]

                sim_vec = torch.nonzero(anchor_data[..., 4] > conf_thres)

                if len(sim_vec) > 0:
                    for x, y in sim_vec:
                        targets.append(anchor_data[x][y])
                        tinfo.append([i, batch_idx, anchor_idx, x, y])

    output = [torch.zeros((0, 10), device=pYolo[0].device)] * batchs
    batch_targets = [None]*batchs
    for idx, target in enumerate(targets):
        info = tinfo[idx]
        batch = info[1]
        feat = pRot[info[0]][batch].permute(1, 2, 0, 3).contiguous()
        anchor_data = feat[info[3]][info[4]]

        idex = anchor_data[:, 0].argmax()
        dir_target = anchor_data[idex]
        obj_merge = torch.cat((target, dir_target)).view(1, -1)
        if batch_targets[batch] is None:
            batch_targets[batch] = []
        batch_targets[batch].append(obj_merge)


    for xi in range(batchs):
        preds = batch_targets[xi]
        if not preds:
            continue
        preds = torch.cat(preds, 0)
        pYolo = preds[:, :5+nc]
        pRot = preds[:, 5+nc:]


        pYolo[:, 5:] *= pYolo[:, 4:5]


        if 1:
            mask_ab = (pRot[..., 3] > ab_thres) & (pRot[..., 4] > ab_thres)
            pRot = pRot[mask_ab]
            pYolo = pYolo[mask_ab]

            if not pYolo.shape[0]:
                continue


        q2 = pRot[:, 1:3]

        if 0:
            q2 = normalize_dir(q2)
        else:
            Lq2 = torch.norm(q2, dim=1, keepdim=True)
            q2 = q2 / (Lq2 + 1e-8)
        if fold_angle == 2:
            q2 = two_fold_angle(q2)


        ab = pRot[:, 3:5]
        center_xy = pYolo[:, :2]
        pts = ab2pts(ab, center_xy, q2)


        conf, clsid = pYolo[:, 5:].max(1, keepdim=True)

        pRot = torch.cat((pts, conf, clsid.float()), 1)

        assert(pRot.shape[0]==pYolo.shape[0])

        max_nms = 30000
        clsid = clsid.squeeze(1)
        keep = []
        for clss in clsid.unique():

            cls_mask = clsid.long() == clss
            cls_idx = torch.nonzero(cls_mask).squeeze(1)
            if(cls_idx.shape[0]>0):
                if(mask_dir is not None and mask_dir!=[] and mask_dir[clss]>0):
                    pcls = pRot[cls_idx, :9]


                    assert(pcls.shape[0] > 0)
                    pcls = pcls.cpu().numpy().astype(np.float64)
                    id_nms = poly_nms_cpu64(pcls, iou_thres)
                else:

                    pcls = pYolo[cls_idx]


                    boxes, scores = pcls[:, :4], pcls[:, 4]
                    box2poly(boxes,pRot,cls_idx)

                    id_nms = torchvision.ops.nms(xywh2xyxy(boxes), scores, iou_thres)
                id_nms = cls_idx[id_nms]
                keep.extend(id_nms)
        output[xi] = pRot[torch.tensor(keep)]


        if(threshs.shape[0]>0):
            rot_obj = torch.tensor([]).to('cuda')
            for t in output[xi]:
                if(t[8] > threshs[int(t[9])]):
                    rot_obj = torch.cat((rot_obj, t.unsqueeze(0)), dim=0)
            output[xi] = rot_obj

    return output


def rot_nms_with_cms(prediction, conf_thres=0.25, iou_thres=0.3, ab_thres=3.0, fold_angle=2, mask_dir=[], cms_s=False, ft_infer=False):
    """Runs Non-Maximum Suppression (NMS) on inference results
    """

    pYolo = prediction[0]
    pRot = prediction[1]
    pCms = [None] * len(pYolo)
    pFt = [None] * len(prediction[0])
    sv_id = 2
    ft_length = 0
    pcms_flag = False

    nc = pYolo.shape[-1] - 5
    if isinstance(conf_thres, float):
        threshs = torch.ones(nc, device=prediction.device) * conf_thres
    else:
        threshs = conf_thres.to(pYolo.device)
        conf_thres = float(conf_thres.min())

    if ft_infer:
        pFt = prediction[sv_id]
        for i, j in enumerate(pFt):
            pFt[i] = j.view(j.shape[0], -1, j.shape[-1])
        pFt = torch.cat(pFt, 1)
        ft_length = pFt.shape[-1]
        sv_id += 1
    pFt = None if isinstance(pFt, list) else pFt
    if cms_s:
        pCms_s = prediction[sv_id]
        sv_id += 1
        for i in range(len(pCms_s)):
            cms_conf, cms_cls = pCms_s[i].max(-1, keepdim=True)
            pCms[i] = torch.cat([cms_conf, cms_cls], dim=-1)
        pcms_flag = True
    if len(prediction) == sv_id+1:
        pCms_v = prediction[sv_id]
        for i in range(len(pCms_v)):
            pCms[i] = pCms_v[i] if not cms_s else torch.cat([pCms[i], pCms_v[i]], dim=-1)
        pcms_flag = True
    pCms = None if pCms[0] is None else pCms
    nl = len(pYolo)
    batchs = pYolo[0].shape[0]
    nc = pYolo[0].shape[-1] - 5


    targets = []
    tinfo = []
    for i in range(nl):

        yolo_map = pYolo[i]
        for batch_idx in range(batchs):
            batch_data = yolo_map[batch_idx]

            na = batch_data.shape[0]
            for anchor_idx in range(na):
                anchor_data = batch_data[anchor_idx]

                sim_vec = torch.nonzero(anchor_data[..., 4] > conf_thres)

                if len(sim_vec) > 0:
                    for x, y in sim_vec:
                        targets.append(anchor_data[x][y])
                        tinfo.append([i, batch_idx, anchor_idx, x, y])
    num_ = 10 if pCms is None else 10 +  + pCms[0].shape[-1]
    output = [torch.zeros((0, num_), device=pYolo[0].device)] * batchs
    batch_targets = [[] for i in range(batchs)]
    for idx, target in enumerate(targets):
        info = tinfo[idx]
        batch = info[1]
        feat = pRot[info[0]][batch].permute(1, 2, 0, 3).contiguous()
        anchor_data = feat[info[3]][info[4]]

        idex = anchor_data[:, 0].argmax()
        dir_target = anchor_data[idex]
        obj_merge = torch.cat((target, dir_target), dim=-1)
        if pFt is not None:
            feat_ft = pFt[info[0]][batch].permute(1, 2, 0, 3).contiguous()
            ft_data = feat_ft[info[3]][info[4]]
            ft_target = ft_data[idex]
            obj_merge = torch.cat((obj_merge, ft_target), dim=-1)
        if pCms is not None:
            feat_cms = pCms[info[0]][batch].permute(1, 2, 0, 3).contiguous()
            cms_data = feat_cms[info[3]][info[4]]
            cms_target = cms_data[idex]
            obj_merge = torch.cat((obj_merge, cms_target), dim=-1)
        obj_merge = obj_merge.view(1, -1)
        batch_targets[batch].append(obj_merge)


    for xi in range(batchs):
        preds = batch_targets[xi]
        if not preds:
            continue
        preds = torch.cat(preds, 0)
        pYolo = preds[:, :5+nc]
        pRot = preds[:, 5+nc:10+nc]
        pFt = preds[:, 10+nc:10+nc+ft_length] if ft_infer else None
        pCms = preds[:, 10+nc+ft_length:] if pcms_flag else None


        pYolo[:, 5:] *= pYolo[:, 4:5]


        if 1:
            mask_ab = (pRot[..., 3] > ab_thres) & (pRot[..., 4] > ab_thres)
            pRot = pRot[mask_ab]
            pYolo = pYolo[mask_ab]
            pCms = pCms[mask_ab] if pCms is not None else None
            pFt = pFt[mask_ab] if pFt is not None else None

            if not pYolo.shape[0]:
                continue


        q2 = pRot[:, 1:3]

        if 0:
            q2 = normalize_dir(q2)
        else:
            Lq2 = torch.norm(q2, dim=1, keepdim=True)
            q2 = q2 / (Lq2 + 1e-8)
        if fold_angle == 2:
            q2 = two_fold_angle(q2)


        ab = pRot[:, 3:5]
        center_xy = pYolo[:, :2]
        pts = ab2pts(ab, center_xy, q2)


        conf, clsid = pYolo[:, 5:].max(1, keepdim=True)

        pRot = torch.cat((pts, conf, clsid.float()), 1)
        if pCms is not None:
            pRot = torch.cat((pRot, pCms), 1)
        if pFt is not None:
            pRot = torch.cat((pRot, pFt), 1)

        assert(pRot.shape[0]==pYolo.shape[0])

        max_nms = 30000
        clsid = clsid.squeeze(1)
        keep = []
        for clss in clsid.unique():

            cls_mask = clsid.long() == clss
            cls_idx = torch.nonzero(cls_mask).squeeze(1)
            if(cls_idx.shape[0]>0):
                if(mask_dir!=[] and mask_dir[clss]>0):
                    pcls = pRot[cls_idx, :9]


                    assert(pcls.shape[0] > 0)
                    pcls = pcls.cpu().numpy().astype(np.float64)
                    id_nms = poly_nms_cpu64(pcls, iou_thres)
                else:

                    pcls = pYolo[cls_idx]


                    boxes, scores = pcls[:, :4], pcls[:, 4]
                    box2poly(boxes,pRot,cls_idx)

                    id_nms = torchvision.ops.nms(xywh2xyxy(boxes), scores, iou_thres)
                id_nms = cls_idx[id_nms]
                keep.extend(id_nms)
        output[xi] = pRot[torch.tensor(keep)]


        if(threshs.shape[0]>0):
            rot_obj = torch.tensor([]).to('cuda')
            for t in output[xi]:
                if(t[8] > threshs[int(t[9])]):
                    rot_obj = torch.cat((rot_obj, t.unsqueeze(0)), dim=0)
            output[xi] = rot_obj

    return output

def non_max_suppression_cms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, return_indices=False):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    indices_grid = torch.arange(0,prediction.shape[1], device=prediction.device).long()

    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres


    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'


    min_wh, max_wh = 2, 4096
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output_indices = [torch.zeros((0, 2), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):


        x = x[xc[xi]]
        grid = indices_grid[xc[xi]]


        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)


        if not x.shape[0]:
            continue


        x[:, 5:] *= x[:, 4:5]


        box = xywh2xyxy(x[:, :4])


        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            grid = grid[i]
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            grid = grid[conf.view(-1) > conf_thres]


        if classes is not None:
            grid = grid[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            grid = grid[x[:, 4].argsort(descending=True)[:max_nms]]
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]


        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):

            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]
        output_indices[xi] = grid[i]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output, output_indices if return_indices else output


def big_nms(det_list, fts, scores, iou_thres):
    if det_list.shape[1]==8:
        y = torch.cat((det_list, scores[:,None]), dim=1)
        y = y.cpu().numpy().astype(np.float64)

        i = poly_nms_cpu64(y, iou_thres)
    elif det_list.shape[1]==4:
        assert det_list.shape[1]==4
        if fts!=[]:
            polys = (ft2pts(fts)).cpu().numpy().astype(np.float64)
            polys2 = np.concatenate([polys, scores.cpu().numpy().astype(np.float64).reshape(-1, 1)], axis=-1)
            i = poly_nms_cpu64(polys2, iou_thres)
        else:
            boxes = det_list[:, :4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
    return i
