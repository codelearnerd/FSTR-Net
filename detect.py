
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights weights/best.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from skimage.feature import peak_local_max
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter
from collections import Counter


import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, print_args, scale_coords, scale_coords_poly,strip_optimizer, xyxy2xywh,
                           xywh2xyxy)
from utils.post_process import rot_nms,rot_nms_with_cms,non_max_suppression,non_max_suppression_ft
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from models.experimental import attempt_load
from tools.plotbox import plot_one_rot_box, plot_one_box_with_cms,plot_one_box_with_ft

from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer,get_source)
from tools.fft.fftinside import render_fourier_curve,render_fourier_curve2
from utils.ft_utils import scale_coords_ft,fft_1
from utils.datasets import get_rgbidx

from scipy.spatial import ConvexHull

from scipy.interpolate import splprep, splev

class MicroPointTracker:

    def __init__(self, max_dist=40):
        self.tracks = {}
        self.next_id = 1
        self.max_dist = max_dist

    def update(self, centers):
        if len(centers) == 0: return []
        centers = np.array(centers)


        if len(self.tracks) == 0:
            out_ids = []
            for pt in centers:
                self.tracks[self.next_id] = tuple(pt)
                out_ids.append(self.next_id)
                self.next_id += 1
            return out_ids

        track_ids = list(self.tracks.keys())
        track_pts = np.array([self.tracks[tid] for tid in track_ids])


        dist_matrix = np.linalg.norm(track_pts[:, None, :] - centers[None, :, :], axis=-1)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        out_ids = [None] * len(centers)
        unmatched_dets = set(range(len(centers)))

        for r, c in zip(row_ind, col_ind):

            if dist_matrix[r, c] < self.max_dist:
                tid = track_ids[r]
                self.tracks[tid] = tuple(centers[c])
                out_ids[c] = tid
                unmatched_dets.remove(c)


        for c in unmatched_dets:
            self.tracks[self.next_id] = tuple(centers[c])
            out_ids[c] = self.next_id
            self.next_id += 1

        return out_ids

def apply_distance_clustering_with_recalc_fourier(det, dist_thresh=80, conf_thresh=0.005, core_conf_thresh=0.10, padding=4):








    det = det[det[:, 4] >= conf_thresh]
    if len(det) == 0:
        return det

    device = det.device
    dtype = det.dtype
    det_np = det.cpu().numpy()


    boxes = det_np[:, :4]
    centers = np.zeros((len(boxes), 2))
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2


    from scipy.sparse.csgraph import connected_components


    dist_matrix = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)


    adj_matrix = dist_matrix < dist_thresh


    _, labels = connected_components(adj_matrix, directed=False)


    clustered_det = []
    num_ft_total = det_np.shape[1] - 6
    num_harmonics = (num_ft_total - 2) // 4

    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        group = det_np[indices]


        cluster_max_conf = np.max(group[:, 4])


        if cluster_max_conf < core_conf_thresh:
            continue

        group_pts = centers[indices]


        if len(indices) == 1:
            clustered_det.append(group[0].tolist())
            continue


        elif len(indices) == 2:
            x1, y1 = np.min(group[:, 0]), np.min(group[:, 1])
            x2, y2 = np.max(group[:, 2]), np.max(group[:, 3])
            cluster_cx, cluster_cy = (x1+x2)/2.0, (y1+y2)/2.0
            scale = (max(x2-x1, y2-y1) / 5.0) * 1.2
            scale = max(1.0, scale)
            best_ft = group[np.argmax(group[:, 4]), 6:].copy()
            best_ft[0], best_ft[1] = cluster_cx, cluster_cy
            best_ft[2:] = best_ft[2:] * scale
            merged_row = [x1-padding, y1-padding, x2+padding, y2+padding, cluster_max_conf, group[0,5]] + best_ft.tolist()
            clustered_det.append(merged_row)
            continue


        try:
            hull = ConvexHull(group_pts)
            hull_pts = group_pts[hull.vertices]

            cluster_cx = np.mean(hull_pts[:, 0])
            cluster_cy = np.mean(hull_pts[:, 1])


            padded_hull = []
            for p in hull_pts:
                dx, dy = p[0] - cluster_cx, p[1] - cluster_cy
                L = np.hypot(dx, dy) + 1e-5
                padded_hull.append([p[0] + (dx/L)*padding, p[1] + (dy/L)*padding])
            padded_hull = np.array(padded_hull)
            padded_hull = np.vstack((padded_hull, padded_hull[0]))


            tck, _ = splprep([padded_hull[:, 0], padded_hull[:, 1]], s=0, per=True)
            xi, yi = splev(np.linspace(0, 1, 128, endpoint=False), tck)


            fft_x = np.fft.fft(xi)
            fft_y = np.fft.fft(yi)
            N = len(xi)

            new_ft = np.zeros(num_ft_total)
            new_ft[0] = cluster_cx
            new_ft[1] = cluster_cy

            for k in range(1, min(num_harmonics + 1, N//2)):
                a_k = (2.0 / N) * np.real(fft_x[k])
                b_k = -(2.0 / N) * np.imag(fft_x[k])
                c_k = (2.0 / N) * np.real(fft_y[k])
                d_k = -(2.0 / N) * np.imag(fft_y[k])

                idx = 2 + (k - 1) * 4
                new_ft[idx]   = a_k
                new_ft[idx+1] = b_k
                new_ft[idx+2] = c_k
                new_ft[idx+3] = d_k

            x1, y1 = np.min(xi), np.min(yi)
            x2, y2 = np.max(xi), np.max(yi)


            merged_row = [x1, y1, x2, y2, cluster_max_conf, group[0, 5]] + new_ft.tolist()
            clustered_det.append(merged_row)

        except Exception as e:

            for pt_row in group:
                clustered_det.append(pt_row.tolist())

    if len(clustered_det) == 0:
        return torch.tensor([], device=device, dtype=dtype)

    return torch.tensor(clustered_det, device=device, dtype=dtype)

def apply_distance_clustering_with_fourier(det, dist_thresh=80, conf_thresh=0.45, ft_padding=1.2):








    det = det[det[:, 4] >= conf_thresh]
    if len(det) == 0:
        return det

    device = det.device
    dtype = det.dtype
    det_np = det.cpu().numpy()


    boxes = det_np[:, :4]
    centers = np.zeros((len(boxes), 2))
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2


    labels = list(range(len(boxes)))
    changed = True
    while changed:
        changed = False
        new_labels = list(labels)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if labels[i] == labels[j]:
                    continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < dist_thresh:
                    target = labels[i]
                    source = labels[j]
                    for k in range(len(boxes)):
                        if labels[k] == source:
                            new_labels[k] = target
                    changed = True
        labels = new_labels


    clustered_det = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        group = det_np[indices]


        if len(indices) == 1:
            clustered_det.append(group[0].tolist())
            continue


        x1, y1 = np.min(group[:, 0]), np.min(group[:, 1])
        x2, y2 = np.max(group[:, 2]), np.max(group[:, 3])

        cluster_w = x2 - x1
        cluster_h = y2 - y1
        cluster_cx = (x1 + x2) / 2.0
        cluster_cy = (y1 + y2) / 2.0

        conf = np.mean(group[:, 4])
        cls = group[0, 5]

        if group.shape[1] > 6:

            best_idx = np.argmax(group[:, 4])
            base_ft = group[best_idx, 6:].copy()


            orig_w = group[best_idx, 2] - group[best_idx, 0]
            orig_h = group[best_idx, 3] - group[best_idx, 1]
            scale_x = cluster_w / (orig_w + 1e-6)
            scale_y = cluster_h / (orig_h + 1e-6)

            scale = max(scale_x, scale_y) * ft_padding


            base_ft[0] = cluster_cx
            base_ft[1] = cluster_cy


            base_ft[2:] = base_ft[2:] * scale


            pad_w = cluster_w * (ft_padding - 1) / 2
            pad_h = cluster_h * (ft_padding - 1) / 2
            merged_row = [x1-pad_w, y1-pad_h, x2+pad_w, y2+pad_h, conf, cls] + base_ft.tolist()
        else:
            merged_row = [x1, y1, x2, y2, conf, cls]

        clustered_det.append(merged_row)

    return torch.tensor(clustered_det, device=device, dtype=dtype)

def apply_distance_clustering(det, dist_thresh=80, conf_thresh=0.45, padding=20):








    det = det[det[:, 4] >= conf_thresh]

    if len(det) == 0:
        return det

    device = det.device
    dtype = det.dtype
    det_np = det.cpu().numpy()

    boxes = det_np[:, :4]


    centers = np.zeros((len(boxes), 2))
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2


    labels = list(range(len(boxes)))
    changed = True
    while changed:
        changed = False
        new_labels = list(labels)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if labels[i] == labels[j]:
                    continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < dist_thresh:
                    target = labels[i]
                    source = labels[j]
                    for k in range(len(boxes)):
                        if labels[k] == source:
                            new_labels[k] = target
                    changed = True
        labels = new_labels


    clustered_det = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        group = det_np[indices]


        if len(indices) == 1:

            clustered_det.append(group[0].tolist())


        else:


            x_min = np.min(group[:, 0]) - padding
            y_min = np.min(group[:, 1]) - padding
            x_max = np.max(group[:, 2]) + padding
            y_max = np.max(group[:, 3]) + padding


            conf = np.mean(group[:, 4])
            cls = group[0, 5]


            if group.shape[1] > 6:


                dummy_ft = np.zeros(group.shape[1] - 6)
                merged_row = [x_min, y_min, x_max, y_max, conf, cls] + dummy_ft.tolist()
            else:
                merged_row = [x_min, y_min, x_max, y_max, conf, cls]

            clustered_det.append(merged_row)

    return torch.tensor(clustered_det, device=device, dtype=dtype)


def cluster_boxes_by_distance(boxes, confs, distance_threshold=80):
    """

    """
    if len(boxes) == 0:
        return [], []


    centers = np.zeros((len(boxes), 2))
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2


    labels = list(range(len(boxes)))


    changed = True
    while changed:
        changed = False
        new_labels = list(labels)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if labels[i] == labels[j]:
                    continue
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < distance_threshold:
                    target_label = labels[i]
                    source_label = labels[j]
                    for k in range(len(boxes)):
                        if labels[k] == source_label:
                            new_labels[k] = target_label
                    changed = True
        labels = new_labels


    unique_labels = np.unique(labels)
    clustered_boxes = []
    clustered_confs = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_group = boxes[indices]
        conf_group = confs[indices]


        x1 = np.min(cluster_group[:, 0])
        y1 = np.min(cluster_group[:, 1])
        x2 = np.max(cluster_group[:, 2])
        y2 = np.max(cluster_group[:, 3])


        avg_conf = np.mean(conf_group)

        clustered_boxes.append([x1, y1, x2, y2])
        clustered_confs.append(avg_conf)

    return np.array(clustered_boxes), np.array(clustered_confs)


def detect_with_cms(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle,mask_dir=[],threshs=torch.zeros(0),dt=[]):

    cms_s = model.cms_s if isinstance(model, DetectMultiBackend) else (model.modules.get_module_byname('SubclassInfer') if hasattr(model, 'module') else model.get_module_byname('SubclassInfer')) is not None
    if dt!=[]:

        t2 = time_sync()
        if training:
            out,train_out = [],[]
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2


        out = rot_nms_with_cms(out, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, cms_s=cms_s)

        dt[2] += time_sync() - t3

        return out,train_out
    else:
        pred = model(im, augment=0, val=True)[0]

        pred = rot_nms_with_cms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, cms_s=cms_s)

        return pred

def detect_ft(model, training, im, augment, conf_thres, iou_thres, dt=[], agnostic=False, ft_infer=False,nms_polygon=1):
    t2 = time_sync()
    if training:
        out,train_out = [],[]
        tmp = model(im, augment=augment)
        for t in tmp:
            out.append(t[0])
            train_out.extend(t[1])
    else:
        out, train_out = model(im, augment=augment, val=True)
    if dt!=[]:
        dt[1] += time_sync() - t2


    pbox_out = []
    for i in out[0]:
        pbox_out.append(i.view(i.shape[0], -1, i.shape[-1]))
    pFt = [None] * len(out[0])
    sv_id = 0
    if ft_infer:
        sv_id = 1
        pFt = out[sv_id]
        for i, j in enumerate(pFt):
            pFt[i] = j.view(j.shape[0], -1, j.shape[-1])
        pFt = torch.cat(pFt, 1)
    pFt = None if isinstance(pFt, list) else pFt


    p2p_preds = None
    if len(out) > 2:
        p2p_preds = out[2]


    pbox_out = torch.cat(pbox_out, 1)

    t3 = time_sync()
    if pFt is None or False:
        out, indices = non_max_suppression(pbox_out, conf_thres, iou_thres, agnostic=agnostic, return_indices=True)
    else:
        out, out_ft, indices = non_max_suppression_ft(pbox_out, pFt, conf_thres if nms_polygon!=2 else 0.05, iou_thres, multi_label=False, agnostic=agnostic, return_indices=True,polygon = nms_polygon==2)

    if pFt is not None:
        pFt = [pFt[i][idx] if idx.shape[0] > 0 else torch.zeros(0, pFt.shape[-1], device=pFt.device) for i, idx in enumerate(indices)]
        for i, pbox in enumerate(out):
            out[i] = torch.cat([pbox, pFt[i]], -1)
    if dt!=[]:
        dt[2] += time_sync() - t3


    return out, p2p_preds


def formation_center_nms(det, ft_start=6, distance_px=30.0, relative_scale=0.35):
    """Suppress duplicate macro anchors using their predicted Fourier centers."""
    if det is None or len(det) <= 1:
        return det
    order = det[:, 4].argsort(descending=True)
    centers = det[:, ft_start:ft_start + 2] if det.shape[1] >= ft_start + 2 else None
    if centers is None or not torch.isfinite(centers).all():
        centers = torch.stack(((det[:, 0] + det[:, 2]) * 0.5,
                               (det[:, 1] + det[:, 3]) * 0.5), dim=1)
    widths = (det[:, 2] - det[:, 0]).clamp(min=1.0)
    heights = (det[:, 3] - det[:, 1]).clamp(min=1.0)

    keep = []
    for idx_tensor in order:
        idx = int(idx_tensor.item())
        duplicate = False
        for kept_idx in keep:
            dist = torch.norm(centers[idx] - centers[kept_idx]).item()
            local_scale = relative_scale * min(
                max(widths[idx].item(), heights[idx].item()),
                max(widths[kept_idx].item(), heights[kept_idx].item()),
            )
            if dist <= max(float(distance_px), float(local_scale)):
                duplicate = True
                break
        if not duplicate:
            keep.append(idx)
    return det[torch.tensor(keep, device=det.device, dtype=torch.long)]


def detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle,mask_dir=[],threshs=torch.zeros(0),dt=[]):

    if dt!=[]:

        t2 = time_sync()
        if training:
            out,train_out = [],[]
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2


        out = rot_nms(out, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir, threshs=threshs)

        dt[2] += time_sync() - t3

        return out,train_out
    else:
        pred = model(im, augment=0, val=True)[0]

        pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir,threshs=threshs)

        return pred

@torch.no_grad()
def run(weights=ROOT / 'weights/best.pt',
        source=ROOT / 'data/images',
        imgsz=[640,640],
        conf_thres=0.1,
        ab_thres=3,
        iou_thres=0.45,
        device='',
        save_txt=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=1,
        half=True,
        plot_label=True,
        dir_line=False,
        fold=2,
        data='',
        mask_dir=[],
        thresh_scale=0.2,
        nms_polygon = 1,
        plot_inside=0,
        cluster_dist = 60,
        formation_nms_dist = 30.0
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)
    else:
        if not os.path.exists(source):
            print(f'\033[91mimages_path:{source} not exist.\033[0m')


    if project!='' and project is not None:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    else:
        weights_path = os.path.dirname(weights)
        train_path = os.path.dirname(weights_path) if os.path.basename(weights_path)=='weights' else weights_path
        assert os.path.isdir(train_path)
        save_dir = increment_path(Path(train_path) / 'detect', exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)


    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)


    stride, names, mask_dir = model.stride, model.names, mask_dir or model.mask_dir


    target_seq_len = 5
    try:
        for m in model.model.modules():
            if hasattr(m, 'seq_len'):
                target_seq_len = m.seq_len
                print(f"[Auto-Detect] Temporal model detected! seq_len = {target_seq_len}")
                break
    except: pass


    pt = True
    half = True
    if pt:
        model.model.half() if half else model.model.float()


    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        bs = 1


    from collections import deque
    input_buffer = deque(maxlen=target_seq_len)
    count_buffer = deque(maxlen=5)


    train_path = Path(weights).parent.parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        threshs = torch.from_numpy(threshs)
    else:
        threshs = torch.ones(len(names)) * conf_thres
    single_txt_conf_thres = float(conf_thres)
    threshs = threshs * thresh_scale
    conf_thres = threshs.to(device)


    if pt and device.type != 'cpu':
        model(torch.zeros(target_seq_len, getattr(model.model, 'ch', 3), *imgsz).to(device).type_as(next(model.model.parameters())))
    dt, seen = [0.0, 0.0, 0.0], 0
    pts = model.pts
    cms_num = model.cms_num
    ft_infer = model.ft_infer
    model_yaml = getattr(getattr(model, 'model', None), 'yaml', {})
    formation_macro_detect = bool(model_yaml.get('formation_macro_detect', False)) if isinstance(model_yaml, dict) else False
    ft_range = [6, 6+model.ft_length] if ft_infer else None
    vid_path, vid_writer = [None] * bs, [None] * bs

    amp_stat = [[0 for i in range((model.ft_length - 2) // 4 + 1)] for j in range(model.model.nc)]


    drone_tracker = MicroPointTracker(max_dist=40)
    unique_tracked_ids = set()

    global_prev_centers = []


    for idimg,(path, im, im0s, vid_cap) in enumerate(dataset):
        video_mode = getattr(dataset, 'mode', 'image') != 'image'

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]


        if len(input_buffer) == 0:
            for _ in range(target_seq_len - 1):
                input_buffer.append(im)
        input_buffer.append(im)


        im_inference = torch.cat(list(input_buffer), dim=0)


        t2 = time_sync()
        dt[0] += t2 - t1


        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        p2p_preds = None
        if pts:

            pred, _ = detect_with_cms(model, False, im_inference, augment, conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir,threshs=threshs, dt=dt)
        else:

            pred, p2p_preds = detect_ft(model, False, im_inference, augment, conf_thres, iou_thres, dt=dt, ft_infer=ft_infer, nms_polygon=nms_polygon)


        if isinstance(pred, list) and len(pred) == target_seq_len:
            pred = [pred[-1]]

        if isinstance(p2p_preds, dict):
            p2p_preds = {k: (v[-1:] if torch.is_tensor(v) and v.shape[0] == target_seq_len else v)
                         for k, v in p2p_preds.items()}


        if True:
            for i, det in enumerate(pred):
                seen += 1

                if webcam:
                    fname, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    fname, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                name = os.path.basename(fname)

                p = Path(fname)
                if p.suffix.lower() == '.bsq':
                    p = p.with_suffix('.jpg')
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem)
                if not os.path.exists(str(save_dir / 'labels')):
                    os.makedirs(str(save_dir / 'labels'))
                s = name
                s += ' %gx%g ' % im.shape[2:]


                micro_points_mapped = []
                micro_box_records = []
                point_boxes_available = False
                bio_density_sum = 0.0
                bio_density_max = 0.0
                bio_density_np = None
                if isinstance(p2p_preds, dict):
                    raw_points = []
                    point_scores = []

                    if "pred_logits" in p2p_preds and "pred_offsets" in p2p_preds:
                        logits = p2p_preds["pred_logits"][i, 0]
                        offsets = p2p_preds["pred_offsets"][i]
                        wh_map = p2p_preds.get("pred_wh", None)
                        wh_map = wh_map[i] if torch.is_tensor(wh_map) else None
                        point_boxes_available = wh_map is not None
                        prob = logits.sigmoid().float().cpu().numpy()
                        bio_density_sum = float(prob.sum())
                        bio_density_max = float(prob.max()) if prob.size else 0.0
                        point_thresh = max(0.05, bio_density_max * 0.12)
                        coordinates = peak_local_max(prob, min_distance=1, threshold_abs=point_thresh)

                        stride_x = im_inference.shape[-1] / max(prob.shape[1], 1)
                        stride_y = im_inference.shape[-2] / max(prob.shape[0], 1)
                        for cy, cx in coordinates:
                            dx = offsets[0, cy, cx].float().item()
                            dy = offsets[1, cy, cx].float().item()
                            abs_x, abs_y = (cx + dx) * stride_x, (cy + dy) * stride_y
                            if wh_map is not None:
                                box_w = max(float(wh_map[0, cy, cx].float().item()) * stride_x, 1.0)
                                box_h = max(float(wh_map[1, cy, cx].float().item()) * stride_y, 1.0)
                                raw_points.append([abs_x - box_w / 2, abs_y - box_h / 2,
                                                   abs_x + box_w / 2, abs_y + box_h / 2])
                            else:
                                raw_points.append([abs_x, abs_y, abs_x, abs_y])
                            point_scores.append(float(prob[cy, cx]))

                    elif "pred_density" in p2p_preds:
                        density = p2p_preds["pred_density"][i, 0].float()
                        bio_density_sum = float(density.sum().detach().cpu())
                        bio_density_max = float(density.max().detach().cpu())
                        bio_density_np = density.detach().cpu().numpy()
                        score_map = density / density.max().clamp(min=1e-6)

                        score_np = score_map.detach().cpu().numpy()
                        threshold = max(0.05, float(score_np.max()) * 0.25)
                        coordinates = peak_local_max(score_np, min_distance=2, threshold_abs=threshold)

                        stride_x = im_inference.shape[-1] / max(score_np.shape[1], 1)
                        stride_y = im_inference.shape[-2] / max(score_np.shape[0], 1)
                        for cy, cx in coordinates:
                            abs_x, abs_y = (cx + 0.5) * stride_x, (cy + 0.5) * stride_y
                            raw_points.append([abs_x, abs_y, abs_x, abs_y])
                            point_scores.append(float(density[cy, cx].detach().cpu()))

                    if len(raw_points) > 0:
                        raw_points_tensor = torch.tensor(raw_points, device=device)
                        mapped_boxes = scale_coords(im_inference.shape[2:], raw_points_tensor, im0.shape).round()
                        for pt_idx, box in enumerate(mapped_boxes):
                            px = int(round((box[0].item() + box[2].item()) * 0.5))
                            py = int(round((box[1].item() + box[3].item()) * 0.5))
                            micro_points_mapped.append((px, py, point_scores[pt_idx]))
                            if point_boxes_available:
                                bw = max(float(box[2].item() - box[0].item()), 1.0)
                                bh = max(float(box[3].item() - box[1].item()), 1.0)
                                micro_box_records.append((
                                    0,
                                    px / max(float(im0.shape[1]), 1.0),
                                    py / max(float(im0.shape[0]), 1.0),
                                    bw / max(float(im0.shape[1]), 1.0),
                                    bh / max(float(im0.shape[0]), 1.0),
                                    point_scores[pt_idx],
                                ))


                nms_count = int(det.shape[0]) if len(det) else 0
                nms_max_conf = float(det[:, 4].max().item()) if len(det) else 0.0
                cluster_count = 0
                label_records = []
                single_records = list(micro_box_records)
                ft_records = []
                polygon_records = []
                bio_points_inside = []
                bio_points_inside_keys = set()
                bio_count_inside_total = 0
                bio_display_points = []
                bio_display_keys = set()
                yolo_points_inside = []
                yolo_points_inside_keys = set()
                yolo_count_inside_total = 0
                density_count_inside_total = 0
                density_raw_inside_total = 0
                single_det_scaled = None
                count_text_items = []
                if len(det):

                    det[:, :4] = scale_coords(im_inference[i].shape[1:], det[:, :4], im0.shape).round()
                    if ft_infer:
                        det[:, ft_range[0]:ft_range[1]] = scale_coords_ft(im_inference[i].shape[1:], det[:, ft_range[0]:ft_range[1]], im0.shape).round()
                    if formation_macro_detect:
                        det = formation_center_nms(
                            det, ft_start=ft_range[0] if ft_infer else 6,
                            distance_px=formation_nms_dist)
                    single_det_scaled = det.clone()
                    W_single, H_single = im0.shape[1], im0.shape[0]
                    for sx1, sy1, sx2, sy2, sconf, scls in single_det_scaled[:, :6]:
                        sx1 = float(sx1.item() if hasattr(sx1, 'item') else sx1)
                        sy1 = float(sy1.item() if hasattr(sy1, 'item') else sy1)
                        sx2 = float(sx2.item() if hasattr(sx2, 'item') else sx2)
                        sy2 = float(sy2.item() if hasattr(sy2, 'item') else sy2)
                        sconf_v = float(sconf.item() if hasattr(sconf, 'item') else sconf)
                        if sconf_v < single_txt_conf_thres:
                            continue
                        scls_id = int(scls.item() if hasattr(scls, 'item') else scls)
                        sx_c = ((sx1 + sx2) * 0.5) / max(float(W_single), 1.0)
                        sy_c = ((sy1 + sy2) * 0.5) / max(float(H_single), 1.0)
                        sbw = (sx2 - sx1) / max(float(W_single), 1.0)
                        sbh = (sy2 - sy1) / max(float(H_single), 1.0)
                        if not point_boxes_available:
                            single_records.append((scls_id, sx_c, sy_c, sbw, sbh, sconf_v))


                    if not formation_macro_detect:
                        det = apply_distance_clustering_with_recalc_fourier(
                                det,
                                dist_thresh=cluster_dist,
                                conf_thresh=0.005,
                                core_conf_thresh=0.02,
                                padding=5
                            )
                    cluster_count = int(det.shape[0]) if len(det) else 0


                    if len(det) > 0:
                        W, H = im0.shape[1], im0.shape[0]
                        if im0.shape[-1] > 3:
                            rgb_idx = list(get_rgbidx(im0.shape[-1]))
                            bgr = np.ascontiguousarray(im0[:, :, rgb_idx[::-1]])
                            im0 = bgr

                        for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                            cls_int = int(cls.item())
                            ft_label = det[j, ft_range[0]:ft_range[1]].cpu() if ft_infer else None

                            if ft_infer and ft_label is not None:
                                assert (ft_label.shape[-1] - 2) % 4 == 0
                                ft_gn = torch.tensor(im0.shape)[[1, 0] + [1, 1, 0, 0] * ((ft_label.shape[-1] - 2) // 4 )].to(ft_label.device)
                                ft_norm = (ft_label / ft_gn).view(-1).tolist()


                                npts = 256
                                x_pol, y_pol = fft_1(ft_norm, npts)
                                polygon_pts = np.stack((np.array(x_pol) * W, np.array(y_pol) * H), axis=1).astype(np.int32)


                                polygon_points_inside = []
                                for pt_x, pt_y, pt_conf in micro_points_mapped:
                                    if cv2.pointPolygonTest(polygon_pts, (float(pt_x), float(pt_y)), False) >= 0:
                                        pt_key = (int(pt_x), int(pt_y))
                                        if formation_macro_detect and pt_key in bio_points_inside_keys:
                                            continue
                                        polygon_points_inside.append((pt_x, pt_y, pt_conf))
                                        if pt_key not in bio_points_inside_keys:
                                            bio_points_inside_keys.add(pt_key)
                                            bio_points_inside.append((pt_x, pt_y, pt_conf))

                                polygon_yolo_points = []
                                if single_det_scaled is not None and len(single_det_scaled):
                                    count_conf_gate = max(0.20, nms_max_conf * 0.25)
                                    for sx1, sy1, sx2, sy2, sconf, scls in single_det_scaled[:, :6]:
                                        if float(sconf.item()) < count_conf_gate:
                                            continue
                                        cx = float(((sx1 + sx2) * 0.5).item())
                                        cy = float(((sy1 + sy2) * 0.5).item())
                                        if cv2.pointPolygonTest(polygon_pts, (cx, cy), False) >= 0:
                                            point_key = (int(round(cx)), int(round(cy)))
                                            polygon_yolo_points.append((point_key[0], point_key[1], float(sconf.item())))

                                if bio_density_np is not None:
                                    h_map, w_map = bio_density_np.shape
                                    polygon_map = polygon_pts.astype(np.float32).copy()
                                    polygon_map[:, 0] = polygon_map[:, 0] * w_map / max(float(W), 1.0)
                                    polygon_map[:, 1] = polygon_map[:, 1] * h_map / max(float(H), 1.0)
                                    polygon_map = np.round(polygon_map).astype(np.int32)
                                    density_mask = np.zeros((h_map, w_map), dtype=np.uint8)
                                    cv2.fillPoly(density_mask, [polygon_map], 1)
                                    raw_density_inside = float((bio_density_np * density_mask).sum())
                                    density_raw_inside_total += int(round(raw_density_inside))
                                    ring_kernel = np.ones((9, 9), dtype=np.uint8)
                                    ring_mask = cv2.dilate(density_mask, ring_kernel, iterations=1) - density_mask
                                    if ring_mask.sum() > 0:
                                        bg_level = float(np.percentile(bio_density_np[ring_mask > 0], 50))
                                    else:
                                        bg_level = float(np.percentile(bio_density_np[density_mask == 0], 20)) if (density_mask == 0).any() else 0.0
                                    foreground_density = np.maximum(bio_density_np - bg_level, 0.0) * density_mask
                                    density_count_inside = int(round(float(foreground_density.sum())))
                                    density_region_max = float((bio_density_np * density_mask).max())
                                else:
                                    density_count_inside = len(polygon_points_inside)
                                    density_region_max = 0.0
                                density_count_inside_total += density_count_inside

                                supported_yolo_points = []
                                if polygon_yolo_points and bio_density_np is not None and density_region_max > 0:
                                    support_gate = 0.20 * density_region_max
                                    for px, py, pconf in polygon_yolo_points:
                                        mx = int(round(px * w_map / max(float(W), 1.0)))
                                        my = int(round(py * h_map / max(float(H), 1.0)))
                                        mx = max(0, min(w_map - 1, mx))
                                        my = max(0, min(h_map - 1, my))
                                        y1, y2 = max(0, my - 1), min(h_map, my + 2)
                                        x1, x2 = max(0, mx - 1), min(w_map, mx + 2)
                                        support = float(bio_density_np[y1:y2, x1:x2].max())
                                        if support >= support_gate:
                                            supported_yolo_points.append((px, py, pconf * support))
                                else:
                                    supported_yolo_points = polygon_yolo_points

                                merged_yolo_points = []
                                merge_radius = 6.0
                                for px, py, pscore in sorted(supported_yolo_points, key=lambda p: p[2], reverse=True):
                                    if all((px - qx) ** 2 + (py - qy) ** 2 > merge_radius ** 2 for qx, qy, _ in merged_yolo_points):
                                        merged_yolo_points.append((px, py, pscore))
                                        point_key = (int(px), int(py))
                                        if point_key not in yolo_points_inside_keys:
                                            yolo_points_inside_keys.add(point_key)
                                            yolo_points_inside.append((px, py, pscore))

                                count_inside_now = (len(polygon_points_inside)
                                                    if formation_macro_detect else density_count_inside)
                                yolo_count_inside_total += len(merged_yolo_points)
                                bio_count_inside_total += count_inside_now

                                if bio_density_np is not None and count_inside_now > 0:
                                    display_map = foreground_density.copy()
                                    selected_display = []
                                    min_dist2 = 2 ** 2
                                    flat_order = np.argsort(display_map.reshape(-1))[::-1]
                                    for flat_idx in flat_order:
                                        if len(selected_display) >= count_inside_now:
                                            break
                                        val = float(display_map.reshape(-1)[flat_idx])
                                        if val <= 0:
                                            break
                                        my, mx = divmod(int(flat_idx), w_map)
                                        px = int(round((mx + 0.5) * W / max(float(w_map), 1.0)))
                                        py = int(round((my + 0.5) * H / max(float(h_map), 1.0)))
                                        if any((px - qx) ** 2 + (py - qy) ** 2 <= min_dist2 for qx, qy, _ in selected_display):
                                            continue
                                        selected_display.append((px, py, val))

                                    for px, py, pscore in selected_display:
                                        point_key = (int(px), int(py))
                                        if point_key not in bio_display_keys:
                                            bio_display_keys.add(point_key)
                                            bio_display_points.append((px, py, pscore))


                                text_x = int(polygon_pts[:, 0].min())
                                text_y = int(polygon_pts[:, 1].min()) - 10
                                text_y = max(20, text_y)
                                count_text_items.append((text_x, text_y, count_inside_now))

                            xyxy_draw = [x_coord.cpu() for x_coord in xyxy]
                            label = '{} {:.2f}'.format(names[cls_int], conf)
                            x1, y1, x2, y2 = [float(v.item() if hasattr(v, 'item') else v) for v in xyxy]
                            x_c = ((x1 + x2) * 0.5) / max(float(W), 1.0)
                            y_c = ((y1 + y2) * 0.5) / max(float(H), 1.0)
                            bw = (x2 - x1) / max(float(W), 1.0)
                            bh = (y2 - y1) / max(float(H), 1.0)
                            label_records.append((cls_int, x_c, y_c, bw, bh, float(conf.item() if hasattr(conf, 'item') else conf), count_inside_now if ft_infer else 0))
                            if ft_infer and ft_label is not None:
                                ft_records.append((cls_int, *ft_norm))
                                polygon_records.append((cls_int, *[coord for point in (polygon_pts.astype(np.float32) / np.array([[max(float(W), 1.0), max(float(H), 1.0)]], dtype=np.float32)) for coord in point]))

                            if not ft_infer:
                                plot_one_box_with_cms(np.array(xyxy_draw), im0,
                                                    color=colors(cls_int%colors.n),
                                                    label=label, line_thickness=line_thickness,
                                                    cms_sub=None, cms_value=None,
                                                    ft_label=ft_label.numpy() if ft_label is not None else None)
                            else:
                                plot_one_box_with_ft(np.array(xyxy_draw), im0,
                                             color=colors(cls_int%colors.n),
                                             label=label, line_thickness=line_thickness,
                                             ft_label=ft_label.numpy(),
                                             amp_stat=amp_stat[cls_int])

                else:
                    label_records = []
                    single_records = list(micro_box_records)
                    ft_records = []
                    polygon_records = []
                    with open(txt_path+'.txt', 'w') as f:
                        pass


                macro_count = len(det) if len(det) else 0
                micro_peak_count = (len(bio_points_inside) if formation_macro_detect else
                                    len(bio_display_points) if ft_infer and len(bio_display_points) else
                                    len(bio_points_inside) if ft_infer else len(micro_points_mapped))
                micro_count_raw = (len(bio_points_inside) if formation_macro_detect else
                                   bio_count_inside_total if ft_infer else len(micro_points_mapped))
                if formation_macro_detect:
                    micro_count = int(micro_count_raw)
                elif ft_infer and macro_count > 0:
                    count_buffer.append(float(micro_count_raw))
                    count_median = float(np.median(np.array(count_buffer, dtype=np.float32)))

                    if abs(micro_count_raw - count_median) <= 3:
                        micro_count = int(round(0.35 * micro_count_raw + 0.65 * count_median))
                    else:
                        micro_count = int(round(micro_count_raw))
                else:
                    micro_count = micro_count_raw

                for text_idx, (text_x, text_y, count_raw) in enumerate(count_text_items):
                    display_count = micro_count if len(count_text_items) == 1 else count_raw
                    cv2.putText(im0, f"Count: {display_count}", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                panel_text = f"Clusters: {macro_count} | BioCount: {micro_count}"
                cv2.rectangle(im0, (10, 10), (450, 60), (0, 0, 0), -1)
                cv2.putText(im0, panel_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if save_txt:
                    with open(txt_path + '.txt', 'w') as f:
                        for cls_id, x_c, y_c, bw, bh, conf_v, count_v in label_records:
                            f.write(('%g ' * 5).rstrip() % (cls_id, x_c, y_c, bw, bh) + '\n')
                    with open(txt_path + '.count', 'w') as f:
                        for idx_rec, (cls_id, x_c, y_c, bw, bh, conf_v, count_v) in enumerate(label_records):
                            f.write(('%g ' * 3).rstrip() % (cls_id, conf_v, count_v) + '\n')
                    if ft_records:
                        with open(txt_path + '.ft', 'w') as f:
                            for rec in ft_records:
                                f.write(('%g ' * len(rec)).rstrip() % rec + '\n')
                    if polygon_records:
                        with open(txt_path + '.pol', 'w') as f:
                            for rec in polygon_records:
                                f.write(('%g ' * len(rec)).rstrip() % rec + '\n')
                    with open(str(save_dir / 'labels' / f'single_{p.stem}.txt'), 'w') as f:
                        for cls_id, x_c, y_c, bw, bh, conf_v in single_records:
                            f.write(('%g ' * 6).rstrip() % (cls_id, x_c, y_c, bw, bh, conf_v) + '\n')


                if not webcam and not video_mode:
                    cv2.imencode(p.suffix, im0)[1].tofile(save_path)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

                debug_det = f"nms={nms_count}, cluster={cluster_count}, max_conf={nms_max_conf:.3f}, bio={micro_count}, bio_raw={micro_count_raw}, yolo_in={yolo_count_inside_total}, density_in={density_count_inside_total}, density_raw={density_raw_inside_total}, bio_peaks={micro_peak_count}, bio_sum={bio_density_sum:.2f}, bio_max={bio_density_max:.4f}, "
                LOGGER.info(f"{s}{debug_det}{'' if len(det) else '(no detections), '}{dt[1]:.3f}ms")


    total_time = dt[1]
    n_img = len(dataset)
    t = tuple(x / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    print(f"total inference time: {total_time}, \
        every inference time:{total_time / n_img}, fps:{1/(total_time / n_img)}\nSaveDir:\n{save_dir}")
    for i in range(model.model.nc):
        print(f'{i:02d}: ',' '.join([f"{amp/(amp_stat[i][-1]+1e-5):.2f}" for amp in amp_stat[i][:-1]]))
