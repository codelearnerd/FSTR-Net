import numpy as np
from utils.geometry_safety import safe_polygon, safe_polygon_iou

try:
    from .poly_cpu import (
        poly_cpu_nms32,
        poly_cpu_iou32,
        poly_cpu_nms32_slow,
        poly_cpu_nms64,
        poly_cpu_iou64,
        poly_cpu_nms64_slow,
    )
except ModuleNotFoundError:
    poly_cpu_nms32 = poly_cpu_iou32 = poly_cpu_nms32_slow = None
    poly_cpu_nms64 = poly_cpu_iou64 = poly_cpu_nms64_slow = None


def _as_polygon(row):
    return safe_polygon(row)


def _poly_iou_fallback(dets1, dets2, dtype):
    dets1 = np.asarray(dets1, dtype=np.float64)
    dets2 = np.asarray(dets2, dtype=np.float64)
    out = np.zeros((dets1.shape[0], dets2.shape[0]), dtype=dtype)
    if dets1.shape[0] == 0 or dets2.shape[0] == 0:
        return out

    polys1 = [_as_polygon(row[:8]) for row in dets1]
    polys2 = [_as_polygon(row[:8]) for row in dets2]
    areas1 = np.asarray([p.area for p in polys1], dtype=np.float64)
    areas2 = np.asarray([p.area for p in polys2], dtype=np.float64)
    for i, p1 in enumerate(polys1):
        if p1.is_empty or areas1[i] <= 0:
            continue
        for j, p2 in enumerate(polys2):
            if p2.is_empty or areas2[j] <= 0:
                continue
            out[i, j] = safe_polygon_iou(p1, p2)
    return out


def _poly_nms_fallback(dets, thresh, dtype):
    dets = np.asarray(dets, dtype=dtype)
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(dtype)

    scores = dets[:, -1]
    polygons = [_as_polygon(row[:8]) for row in dets]
    valid = np.asarray([not poly.is_empty for poly in polygons]) & np.isfinite(scores)
    valid_ids = np.flatnonzero(valid)
    order = valid_ids[scores[valid_ids].argsort()[::-1]]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.asarray([safe_polygon_iou(polygons[i], polygons[j]) for j in order[1:]])
        order = order[1:][ious <= thresh]
    return np.asarray(keep, dtype=dtype)

def poly_nms_cpu32(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float32)
    if poly_cpu_nms32 is None:
        return _poly_nms_fallback(dets, thresh, np.float32)
    return poly_cpu_nms32(dets, thresh)

def poly_nms_cpu32_slow(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float32)
    if poly_cpu_nms32_slow is None:
        return _poly_nms_fallback(dets, thresh, np.float32)
    return poly_cpu_nms32_slow(dets, thresh)

def poly_iou_cpu32(dets1, dets2):
    if dets1.shape[0] == 0 or dets2.shape[0] == 0:
        return np.zeros([dets1.shape[0], dets2.shape[0]]).astype(np.float32)
    if poly_cpu_iou32 is None:
        return _poly_iou_fallback(dets1, dets2, np.float32)
    return poly_cpu_iou32(dets1, dets2)

def poly_nms_cpu64(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float64)
    if poly_cpu_nms64 is None:
        return _poly_nms_fallback(dets, thresh, np.float64)
    return poly_cpu_nms64(dets, thresh)

def poly_nms_cpu64_slow(dets, thresh):
    if dets.shape[0] == 0:
        return np.zeros([0]).astype(np.float64)
    if poly_cpu_nms64_slow is None:
        return _poly_nms_fallback(dets, thresh, np.float64)
    return poly_cpu_nms64_slow(dets, thresh)

def poly_iou_cpu64(dets1, dets2):
    if dets1.shape[0] == 0 or dets2.shape[0] == 0:
        return np.zeros([dets1.shape[0], dets2.shape[0]]).astype(np.float64)
    if poly_cpu_iou64 is None:
        return _poly_iou_fallback(dets1, dets2, np.float64)
    return poly_cpu_iou64(dets1, dets2)

__all__ = ['poly_nms_cpu32', 'poly_iou_cpu32', 'poly_nms_cpu32_slow', 'poly_nms_cpu64', 'poly_iou_cpu64', 'poly_nms_cpu64_slow']
