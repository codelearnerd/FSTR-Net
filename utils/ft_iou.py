from utils.geometry_safety import safe_polygon, safe_polygon_iou, record_invalid_geometry
import os
import numpy as np
import torch
import warnings


def polygon_iou(poly1_pts, poly2_pts):
    return safe_polygon_iou(safe_polygon(poly1_pts), safe_polygon(poly2_pts))


def read_ft(ft_file):
    if os.path.exists(ft_file):
        with open(ft_file) as f:
            l_ft = [x.split() for x in f.read().strip().splitlines() if len(x)]
            l_ft = np.array(l_ft, dtype=np.float32)
    else:
        return None
    return l_ft

def get_curve(ft_label,num=200):
    theta_fine = np.linspace(0, 2*np.pi, num)
    an,bn,cn,dn = np.split(ft_label[2:].reshape(-1, 4), 4, axis=-1)
    an,bn,cn,dn = np.insert(an,0,ft_label[0]),np.insert(bn,0,0),np.insert(cn,0,ft_label[1]),np.insert(dn,0,0)

    def ft2xy(an,bn,cn,dn,theta_fine,term):
        term = an.shape[0] if term<=0 else term
        x_approx = sum([an[i]*np.cos(i*theta_fine) + bn[i]*np.sin(i*theta_fine) for i in range(term)])
        y_approx = sum([cn[i]*np.cos(i*theta_fine) + dn[i]*np.sin(i*theta_fine) for i in range(term)])
        return x_approx,y_approx

    x_approx,y_approx = ft2xy(an,bn,cn,dn,theta_fine,0)
    contour_approx = np.array(list(zip(x_approx, y_approx))).reshape((-1, 2))

    return contour_approx

def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (np.ndarray[N, 4])
        box2 (np.ndarray[M, 4])

    Returns:
        iou (np.ndarray[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)


    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])

    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]


    union = area1[:, None] + area2 - inter


    iou = inter / np.maximum(union, 1e-7)
    return iou

def polygon_iou_ft(ft_1, ft_2, classes_1=None, classes_2=None):
    if len(ft_1) == 0 or len(ft_2) == 0:
        return np.zeros((len(ft_1), len(ft_2)), dtype=np.float64)
    ft_1 = ft_1.cpu().numpy().astype(np.float64)
    ft_2 = ft_2.cpu().numpy().astype(np.float64)
    ft_1_curve = np.stack([get_curve(ft) for ft in ft_1])
    ft_2_curve = np.stack([get_curve(ft) for ft in ft_2])
    ft_1_xyxy = np.concatenate([np.amin(ft_1_curve, axis=1), np.amax(ft_1_curve, axis=1)], axis=-1)
    ft_2_xyxy = np.concatenate([np.amin(ft_2_curve, axis=1), np.amax(ft_2_curve, axis=1)], axis=-1)

    bbox_iou = box_iou(ft_1_xyxy, ft_2_xyxy)
    invalid = ~np.isfinite(ft_1).all(axis=1)
    invalid2 = ~np.isfinite(ft_2).all(axis=1)
    if invalid.any() or invalid2.any():
        record_invalid_geometry('nonfinite_fourier_iou', invalid.sum() + invalid2.sum())
    bbox_iou[invalid, :] = 0
    bbox_iou[:, invalid2] = 0
    bbox_iou = np.nan_to_num(bbox_iou, nan=0.0, posinf=0.0, neginf=0.0)


    if (classes_1 is not None) and (classes_2 is not None):
        assert classes_1.shape[0] == ft_1.shape[0]
        assert classes_2.shape[0] == ft_2.shape[0]
        bbox_iou[(classes_1.reshape(-1, 1) != classes_2.reshape(1, -1)).detach().cpu().numpy()] = 0.
    for i in range(bbox_iou.shape[0]):
        for j in range(bbox_iou.shape[1]):
            if bbox_iou[i, j] > 0:
                bbox_iou[i, j] = polygon_iou(ft_1_curve[i], ft_2_curve[j])
    return bbox_iou


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    return (np.concatenate if isinstance(x, np.ndarray) else torch.cat)((xy - wh, xy + wh), -1)


def polygon_iou_hv(ft_1, hv_2, classes_1=None, classes_2=None, xywh=False):
    ft_1 = ft_1.cpu().numpy().astype(np.float64)
    if xywh:
        hv_2 = xywh2xyxy(hv_2)
    hv_2 = hv_2.cpu().numpy().astype(np.float64)

    ft_1_curve = np.stack([get_curve(ft) for ft in ft_1])

    ft_1_xyxy = np.concatenate([np.amin(ft_1_curve, axis=1), np.amax(ft_1_curve, axis=1)], axis=-1)

    hv_2_curve = np.stack([np.array(hv[[0,1,0,3,2,3,2,1]]).reshape(-1, 2) for hv in hv_2])

    bbox_iou = box_iou(ft_1_xyxy, hv_2)


    if (classes_1 is not None) and (classes_2 is not None):
        assert classes_1.shape[0] == ft_1.shape[0]
        assert classes_2.shape[0] == hv_2_curve.shape[0]
        bbox_iou[(classes_1.reshape(-1, 1) != classes_2.reshape(1, -1)).detach().cpu().numpy()] = 0.
    for i in range(bbox_iou.shape[0]):
        for j in range(bbox_iou.shape[1]):
            if bbox_iou[i, j] > 0:
                bbox_iou[i, j] = polygon_iou(ft_1_curve[i], hv_2_curve[j])
    return bbox_iou

def polygon_nms_ft(fts, scores, cls, iou_thresh=0.45, cls_offset=0, h_thresh_dec=0.6,dif_cls_iou_thresh=1.1):

    finite = torch.isfinite(fts).all(1) & torch.isfinite(scores)
    source_ids = torch.nonzero(finite, as_tuple=False).flatten().cpu().numpy()
    if not bool(finite.all()):
        record_invalid_geometry('nonfinite_fourier_nms', int((~finite).sum()))
    fts, scores = fts[finite], scores[finite]
    cls = cls.detach().cpu().numpy() if torch.is_tensor(cls) else np.asarray(cls)
    cls = cls[source_ids]
    if not len(source_ids):
        return np.empty(0, dtype=int)
    fts = fts.cpu().numpy().astype(np.float64)
    if cls_offset:
        fts[:, :2] = fts[:, :2] + cls[:,None] * 4096
    fts_curve = np.stack([get_curve(ft) for ft in fts])
    fts_xyxy = np.concatenate([np.amin(fts_curve, axis=1), np.amax(fts_curve, axis=1)], axis=-1)

    order = scores.cpu().numpy().argsort()[::-1].copy()
    keep = []
    while order.size > 0:
        order_i = order[0]
        keep.append(order_i)
        cls_i = cls[order_i]

        if order.size == 1:
            break

        next_order = order[1:]
        hbb_ord_iou = box_iou(fts_xyxy[order_i:order_i+1], fts_xyxy[next_order])[0]
        cls_ord = cls[next_order]
        assert hbb_ord_iou.shape[0]==cls_ord.shape[0]
        h_ord_ids = np.where(hbb_ord_iou > h_thresh_dec * iou_thresh)[0]
        ids = next_order[h_ord_ids]

        iou = [polygon_iou(fts_curve[order_i], fts_curve[tmp]) for tmp in ids]
        hbb_ord_iou[h_ord_ids] = np.array(iou)

        same_cls_mask = cls_ord==cls_i
        if dif_cls_iou_thresh >= 1.0:

            keep_mask = (~same_cls_mask) | (hbb_ord_iou <= iou_thresh)
        else:
            diff_cls_mask = ~same_cls_mask

            keep_mask = np.ones_like(hbb_ord_iou, dtype=bool)

            keep_mask[same_cls_mask] = hbb_ord_iou[same_cls_mask] <= iou_thresh

            keep_mask[diff_cls_mask] = hbb_ord_iou[diff_cls_mask] <= dif_cls_iou_thresh


        order = next_order[keep_mask]

    return source_ids[np.asarray(keep, dtype=int)]

def polygon_nms_ft_ok(fts, scores, cls, iou_thresh=0.45):
    fts = fts.cpu().numpy().astype(np.float64)
    scores = scores.cpu().numpy()

    fts_curve = np.stack([get_curve(ft) for ft in fts])
    fts_xyxy = np.concatenate([
        np.amin(fts_curve, axis=1), np.amax(fts_curve, axis=1)
    ], axis=-1)

    order = scores.argsort()[::-1].copy()
    keep = []

    while order.size > 0:
        order_i = order[0]
        keep.append(order_i)
        cls_i = cls[order_i]

        if order.size == 1:
            break

        next_order = order[1:]
        hbb_ord_iou = box_iou(fts_xyxy[order_i:order_i+1], fts_xyxy[next_order])[0]
        cls_ord = cls[next_order]

        h_ord_ids = np.where(hbb_ord_iou > 0)[0]
        ids = next_order[h_ord_ids]

        iou = [polygon_iou(fts_curve[order_i], fts_curve[tmp]) for tmp in ids]
        hbb_ord_iou[h_ord_ids] = np.array(iou)


        keep_mask = (cls_ord != cls_i) | (hbb_ord_iou <= iou_thresh)
        order = next_order[keep_mask]

    return np.asarray(keep, dtype=int)

from DOTA_devkit.polyiou_cpu import poly_iou_cpu64
from utils.ft_utils import ft2pts
def box_iou_ft(coef1s, coef2s, n=15):


    boxes1 = ft2pts(coef1s).cpu().numpy().astype(np.float64)
    boxes2 = ft2pts(coef2s).cpu().numpy().astype(np.float64)

    return torch.from_numpy(poly_iou_cpu64(boxes1, boxes2)).to(coef1s.device).float()


if __name__ == '__main__':
    from pathlib import Path
    ft_file = Path('~/workspace/datas/coco2017/train/labels/000000000009.ft').expanduser().resolve()
    img_file = Path('~/workspace/datas/coco2017/train/images/000000000009.jpg').expanduser().resolve()
    from PIL import Image, ImageDraw
    import cv2
    img = Image.open(img_file)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    ft = read_ft(ft_file)

    ious = polygon_iou_ft(ft[:, 1:], ft[:, 1:], ft[:, :1], ft[:, :1])
    print()


