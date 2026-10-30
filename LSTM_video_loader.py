import glob
import os
import cv2
import random
import math
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from utils.augmentations import augment_hsv, augment_ft
from utils.general import xywhn2xyxy, xyxy2xywhn, xyxy2xywh


def ftnorm2ft(k, w, h):
    k_new = k.clone() if isinstance(k, torch.Tensor) else k.copy()
    n_coefs = k.shape[1] // 4
    k_reshaped = k_new.reshape(-1, n_coefs, 4)
    k_reshaped[:, :, 0] *= w
    k_reshaped[:, :, 1] *= w
    k_reshaped[:, :, 2] *= h
    k_reshaped[:, :, 3] *= h
    return k_reshaped.reshape(k.shape)

def ft2ftnorm(k, w, h):
    k_new = k.clone() if isinstance(k, torch.Tensor) else k.copy()
    n_coefs = k.shape[1] // 4
    k_reshaped = k_new.reshape(-1, n_coefs, 4)
    k_reshaped[:, :, 0] /= w
    k_reshaped[:, :, 1] /= w
    k_reshaped[:, :, 2] /= h
    k_reshaped[:, :, 3] /= h
    return k_reshaped.reshape(k.shape)

def get_affine_matrix(target_size, degrees=0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0):
    width, height = target_size
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2
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
    T[0, 2] = width / 2 + random.uniform(-translate, translate) * width
    T[1, 2] = height / 2 + random.uniform(-translate, translate) * height
    M = T @ S @ R @ P @ C
    return M, s

def mixup_video(im_seq1, labels_seq1, im_seq2, labels_seq2):
    """ Mixup ( 5 )"""
    r = np.random.beta(32.0, 32.0)
    new_im_seq = []
    new_labels_seq = []
    for t in range(len(im_seq1)):
        new_im = (im_seq1[t] * r + im_seq2[t] * (1 - r)).astype(np.uint8)
        new_im_seq.append(new_im)
        new_labels_seq.append(np.concatenate((labels_seq1[t], labels_seq2[t]), 0))
    return new_im_seq, new_labels_seq

def random_perspective_video(im_seq, targets_seq=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0), ft_coef=0, end_i=5):
    """"""
    height = im_seq[0].shape[0] + border[0] * 2
    width = im_seq[0].shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -im_seq[0].shape[1] / 2
    C[1, 2] = -im_seq[0].shape[0] / 2
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

    new_im_seq = []
    for im in im_seq:
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
            if perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        new_im_seq.append(im)

    new_targets_seq = []
    for targets in targets_seq:
        n = len(targets)
        if n:
            new_targets = targets.copy()


            transformed_ft = None
            if ft_coef > 0:
                ft_length = 2 + 4 * ft_coef
                transformed_ft = augment_ft(
                    targets[:, end_i:end_i + ft_length].copy(),
                    M[:2].astype(np.float32),
                )
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new_targets[:, 1:5] = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            new_targets[:, [1, 3]] = new_targets[:, [1, 3]].clip(0, width)
            new_targets[:, [2, 4]] = new_targets[:, [2, 4]].clip(0, height)
            w_new = new_targets[:, 3] - new_targets[:, 1]
            h_new = new_targets[:, 4] - new_targets[:, 2]
            mask = (w_new > 1) & (h_new > 1)
            new_targets = new_targets[mask]

            if transformed_ft is not None and len(new_targets) > 0:
                new_targets[:, end_i:end_i + transformed_ft.shape[1]] = transformed_ft[mask]

            new_targets_seq.append(new_targets)
        else:
            new_targets_seq.append(targets)

    return new_im_seq, new_targets_seq

def compute_coefficients(xy, terms=8, interp=False, dist=True, line=False):
    if len(xy) == 0: return [0.0] * (2 + 4 * terms)
    x = np.array(xy[0::2])
    y = np.array(xy[1::2])
    if interp:
        if dist:
            if not line:
                x = np.concatenate([x, [x[0]]], dtype=np.float32)
                y = np.concatenate([y, [y[0]]], dtype=np.float32)
            else:
                x = np.concatenate([x, x[::-1]], dtype=np.float32)
                y = np.concatenate([y, y[::-1]], dtype=np.float32)
            distances = np.concatenate([np.zeros(1), np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)], dtype=np.float32)
            distances = np.cumsum(distances)
            if distances[-1] > 0: ori = distances / distances[-1]
            else: ori = np.linspace(0, 1, len(distances))
        else:
            ori = np.linspace(0, 1, x.shape[0], endpoint=True)
        gap = np.linspace(0, 1, max(terms*2, x.shape[0] - 1), endpoint=False)
        x = np.interp(gap, ori, x)
        y = np.interp(gap, ori, y)
        N = x.shape[0]
    else:
        N = x.shape[0]
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    a0 = 1./N * sum(x); c0 = 1./N * sum(y)
    an, bn, cn, dn = [np.zeros(1 + terms) for i in range(4)]
    for k in range(1, (N // 2) + 1):
        if k > terms: break
        an[k] = 2./N * sum(x * np.cos(k*t))
        bn[k] = (2./N * sum(x * np.sin(k*t))) if not line else 0
        cn[k] = 2./N * sum(y * np.cos(k*t))
        dn[k] = (2./N * sum(y * np.sin(k*t))) if not line else 0
    list_coef = [a0, c0]
    for k in range(1, terms + 1):
        if k < len(an): list_coef.extend([an[k], bn[k], cn[k], dn[k]])
        else: list_coef.extend([0, 0, 0, 0])
    return list_coef


class VideoFourierDataset(Dataset):


    _has_logged_network_input = False

    def __init__(self, path, img_size=640, seq_length=5, ft_coef=8, stride=1, augment=False, hyp=None):
        self.img_size = img_size
        self.seq_len = seq_length
        self.ft_coef = ft_coef
        self.stride = stride
        self.augment = augment
        self.hyp = hyp
        self.macro_ft_only = bool((hyp or {}).get('formation_macro_targets', False))

        self.end_i = 5
        self.ft_length = ft_coef * 4 + 2 if ft_coef > 0 else 0
        self.label_length = 5 + self.ft_length

        self.mosaic = self.augment
        if isinstance(self.img_size, int):
            self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        else:
            self.mosaic_border = [-self.img_size[0] // 2, -self.img_size[1] // 2]

        p = str(Path(path).absolute())
        if not os.path.exists(p): raise Exception(f"Path not found: {p}")
        self.video_dirs = sorted(glob.glob(os.path.join(p, '*')))
        self.video_dirs = [x for x in self.video_dirs if os.path.isdir(x)]

        self.all_imgs = []
        self.clips = []

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        for v_dir in self.video_dirs:
            raw_imgs = glob.glob(os.path.join(v_dir, '*.jpg')) +\
                       glob.glob(os.path.join(v_dir, '*.png')) +\
                       glob.glob(os.path.join(v_dir, '*.bmp'))
            try:
                raw_imgs.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except:
                raw_imgs.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

            if len(raw_imgs) < self.seq_len: continue

            start_idx = len(self.all_imgs)
            self.all_imgs.extend(raw_imgs)

            num_clips = (len(raw_imgs) - self.seq_len) // self.stride + 1
            for i in range(num_clips):
                self.clips.append(start_idx + i * self.stride)

        self.labels = [None] * len(self.all_imgs)
        self.cm = None
        self._cache_labels_and_calc_cm()

        self.n = len(self.clips)
        self.indices = range(self.n)

    def _cache_labels_and_calc_cm(self):
        fourier_coefs_list = []
        for global_idx in tqdm(range(len(self.all_imgs)), desc="Caching Labels (All Frames)"):
            img_path = self.all_imgs[global_idx]

            base_path = img_path.replace('images', 'labels').rsplit('.', 1)[0]
            cluster_txt_path = base_path + '.txt'
            ft_path = base_path + '.ft'

            dir_name = os.path.dirname(base_path)
            file_name = os.path.basename(base_path)
            single_txt_path = os.path.join(dir_name, f"single_{file_name}.txt")

            cluster_data, single_data, ft_data = [], [], []
            if os.path.exists(cluster_txt_path):
                with open(cluster_txt_path) as f:
                    cluster_data = [list(map(float, x.strip().split())) for x in f.readlines()]
            if os.path.exists(single_txt_path):
                with open(single_txt_path) as f:
                    single_data = [list(map(float, x.strip().split())) for x in f.readlines()]
            if os.path.exists(ft_path):
                with open(ft_path) as f:
                    ft_data = [list(map(float, x.strip().split())) for x in f.readlines()]

            if len(cluster_data) == 0 and len(single_data) == 0:
                self.labels[global_idx] = np.zeros((0, 5 + 2 + self.ft_coef * 4), dtype=np.float32)
                continue

            processed_labels = []

            for i, row in enumerate(cluster_data):
                if len(row) < 5: continue
                _, cx, cy, w, h = row[:5]
                cls_id = 1.0
                coefs = []
                needed_len = 2 + 4 * self.ft_coef

                if i < len(ft_data):
                    raw_ft = ft_data[i]
                    if len(raw_ft) > 1: raw_ft = raw_ft[1:]
                    if len(raw_ft) >= needed_len:
                        coefs = raw_ft[:needed_len]
                    else:
                        coefs = raw_ft + [0.0] * (needed_len - len(raw_ft))
                    coefs[0] = cx
                    coefs[1] = cy
                else:
                    if len(row) > 6: points = row[5:]
                    else:
                        w2, h2 = w/2, h/2
                        points = [cx-w2, cy-h2, cx+w2, cy-h2, cx+w2, cy+h2, cx-w2, cy+h2]
                    coefs = compute_coefficients(points, terms=self.ft_coef, interp=True)

                coefs = np.array(coefs, dtype=np.float32)
                fourier_coefs_list.append(coefs)
                processed_labels.append(np.concatenate(([cls_id, cx, cy, w, h], coefs)))

            for i, row in enumerate(single_data):
                if len(row) < 5: continue
                _, cx, cy, w, h = row[:5]
                cls_id = 0.0

                w2, h2 = w/2, h/2
                points = [cx-w2, cy-h2, cx+w2, cy-h2, cx+w2, cy+h2, cx-w2, cy+h2]
                coefs = compute_coefficients(points, terms=self.ft_coef, interp=True)

                coefs = np.array(coefs, dtype=np.float32)
                if not self.macro_ft_only:
                    fourier_coefs_list.append(coefs)
                processed_labels.append(np.concatenate(([cls_id, cx, cy, w, h], coefs)))

            if len(processed_labels) > 0:
                self.labels[global_idx] = np.stack(processed_labels)
            else:
                self.labels[global_idx] = np.zeros((0, 5 + 2 + self.ft_coef * 4), dtype=np.float32)

        if len(fourier_coefs_list) > 0:
            all_coefs = np.vstack(fourier_coefs_list)
            shape_coefs = all_coefs[:, 2:]
            shape_coefs = [x for x in shape_coefs if x is not None and len(x) > 0]
            if len(shape_coefs) > 0:
                with np.errstate(all='ignore'): std_all = np.std(shape_coefs, axis=0)
                std_all[std_all == 0] = 1e-8
            else:
                std_all = np.ones(self.ft_coef * 4)

            if std_all.shape[0] % 4 == 0: self.cm = std_all.reshape(-1, 4).mean(axis=1)
            else: self.cm = std_all
            if self.cm.sum() == 0: self.cm = np.ones_like(self.cm) * 0.01
        else:
            self.cm = np.zeros(self.ft_coef, dtype=np.float32)

    def __len__(self):
        return len(self.clips)

    def load_image_resized(self, index, t):
        global_idx = self.clips[index]
        path = self.all_imgs[global_idx + t]
        img = cv2.imread(path)

        s = self.img_size
        if isinstance(s, (list, tuple)): s = max(s)

        if img is None:
            img = np.zeros((s, s, 3), dtype=np.uint8)
            return img, (s, s), (0, 0)

        r = min(s / img.shape[0], s / img.shape[1])
        h0, w0 = img.shape[:2]
        if r != 1:
            img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        top = int(round((s - h) / 2 - 0.1))
        bottom = int(round((s - h) / 2 + 0.1))
        left = int(round((s - w) / 2 - 0.1))
        right = int(round((s - w) / 2 + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, (h, w), (left, top)

    def load_mosaic(self, index):
        s = self.img_size
        if isinstance(s, (list, tuple)): s = max(s)

        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in [-s // 2, -s // 2]]
        indices = [index] + random.choices(range(len(self.clips)), k=3)
        random.shuffle(indices)

        imgs_seq = []
        labels_seq = []
        for t in range(self.seq_len):
            img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
            labels4_t = []
            for i, idx in enumerate(indices):
                img, (h, w), (inner_padw, inner_padh) = self.load_image_resized(idx, t)
                img_h, img_w = img.shape[:2]

                if i == 0:
                    x1a, y1a, x2a, y2a = max(xc - img_w, 0), max(yc - img_h, 0), xc, yc
                    x1b, y1b, x2b, y2b = img_w - (x2a - x1a), img_h - (y2a - y1a), img_w, img_h
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, max(yc - img_h, 0), min(xc + img_w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, img_h - (y2a - y1a), min(img_w, x2a - x1a), img_h
                elif i == 2:
                    x1a, y1a, x2a, y2a = max(xc - img_w, 0), yc, xc, min(s * 2, yc + img_h)
                    x1b, y1b, x2b, y2b = img_w - (x2a - x1a), 0, img_w, min(y2a - y1a, img_h)
                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + img_w, s * 2), min(s * 2, yc + img_h)
                    x1b, y1b, x2b, y2b = 0, 0, min(img_w, x2a - x1a), min(y2a - y1a, img_h)

                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                labels = self.labels[self.clips[idx] + t].copy()
                if labels.size > 0:
                    from utils.general import xywhn2xyxy
                    labels[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i], w, h, padw + inner_padw, padh + inner_padh)
                    if self.ft_coef > 0:
                        labels[:, self.end_i] = labels[:, self.end_i] * w + inner_padw + padw
                        labels[:, self.end_i + 1] = labels[:, self.end_i + 1] * h + inner_padh + padh
                        labels[:, self.end_i+2 : self.end_i+self.ft_length] = ftnorm2ft(labels[:, self.end_i+2 : self.end_i+self.ft_length], w, h)
                    labels4_t.append(labels)

            imgs_seq.append(img4)
            if len(labels4_t) > 0:
                labels4_t = np.concatenate(labels4_t, 0)
                np.clip(labels4_t[:, 1:5], 0, 2 * s, out=labels4_t[:, 1:5])
            else:
                labels4_t = np.zeros((0, self.label_length), dtype=np.float32)
            labels_seq.append(labels4_t)

        return imgs_seq, labels_seq, (s, s)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']

        if mosaic:
            img_seq, labels_seq, _ = self.load_mosaic(index)
            if random.random() < hyp['mixup']:
                img_seq2, labels_seq2, _ = self.load_mosaic(random.randint(0, self.n - 1))
                img_seq, labels_seq = mixup_video(img_seq, labels_seq, img_seq2, labels_seq2)
        else:
            img_seq = []
            labels_seq = []
            global_idx = self.clips[index]
            for t in range(self.seq_len):
                img, (h0, w0), (padw, padh) = self.load_image_resized(index, t)
                img_seq.append(img)

                h, w = img_seq[0].shape[:2]
                labels = self.labels[global_idx + t].copy()
                if labels.size:
                    from utils.general import xywhn2xyxy
                    labels[:, 1:self.end_i] = xywhn2xyxy(labels[:, 1:self.end_i], w0, h0, padw=padw, padh=padh)
                    if self.ft_coef > 0:
                        labels[:, self.end_i] = labels[:, self.end_i] * w0 + padw
                        labels[:, self.end_i + 1] = labels[:, self.end_i + 1] * h0 + padh
                        labels[:, self.end_i+2 : self.end_i+self.ft_length] = ftnorm2ft(labels[:, self.end_i+2 : self.end_i+self.ft_length], w0, h0)
                labels_seq.append(labels)

        if self.augment:
            img_seq, labels_seq = random_perspective_video(
                img_seq, labels_seq,
                degrees=hyp.get('degrees', 0), translate=hyp.get('translate', 0),
                scale=hyp.get('scale', 0), shear=hyp.get('shear', 0),
                perspective=hyp.get('perspective', 0), border=self.mosaic_border if mosaic else (0, 0),
                ft_coef=self.ft_coef, end_i=self.end_i
            )

            if random.random() < hyp['fliplr']:
                img_seq = [cv2.flip(im, 1) for im in img_seq]
                for t in range(self.seq_len):
                    if len(labels_seq[t]) > 0:
                        width = img_seq[0].shape[1]
                        labels_seq[t][:, [1, 3]] = width - labels_seq[t][:, [3, 1]]
                        if self.ft_coef > 0:
                            ft_length = 2 + 4 * self.ft_coef
                            flip_matrix = np.array(
                                [[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0]],
                                dtype=np.float32,
                            )
                            labels_seq[t][:, self.end_i:self.end_i + ft_length] = augment_ft(
                                labels_seq[t][:, self.end_i:self.end_i + ft_length].copy(),
                                flip_matrix,
                            )

        w, h = img_seq[0].shape[1], img_seq[0].shape[0]
        labels_out_list = []
        for t in range(self.seq_len):
            labels = labels_seq[t]
            nl = len(labels)
            if nl:
                from utils.general import xyxy2xywhn
                labels[:, 1:self.end_i] = xyxy2xywhn(labels[:, 1:self.end_i], w=w, h=h, clip=False, eps=1E-3)
                if self.ft_coef > 0:
                    labels[:, self.end_i] /= w; labels[:, self.end_i + 1] /= h
                    labels[:, self.end_i+2 : self.end_i+self.ft_length] = ft2ftnorm(labels[:, self.end_i+2 : self.end_i+self.ft_length], w=w, h=h)

                out_t = torch.zeros((nl, self.label_length + 1))
                out_t[:, 0] = t
                out_t[:, 1:] = torch.from_numpy(labels)
                labels_out_list.append(out_t)

        if len(labels_out_list) > 0:
            labels_out = torch.cat(labels_out_list, 0)
        else:
            labels_out = torch.zeros((0, self.label_length + 1))


        final_imgs = [im[:, :, ::-1].transpose(2, 0, 1) for im in img_seq]
        final_imgs = np.stack(final_imgs)
        final_imgs = np.ascontiguousarray(final_imgs, dtype=np.uint8)


        path_seq = [self.all_imgs[self.clips[index] + t] for t in range(self.seq_len)]


        if not hasattr(self, 'has_saved_debug_vis'):
            self.has_saved_debug_vis = False

        if not self.has_saved_debug_vis and len(labels_out) > 0:
            self.has_saved_debug_vis = True
            debug_dir = "debug_dataloader_clean_fourier"
            os.makedirs(debug_dir, exist_ok=True)
            for t_idx in range(self.seq_len):
                im_chw = final_imgs[t_idx]
                im_hwc = im_chw.transpose(1, 2, 0)
                im_bgr = im_hwc[:, :, ::-1].copy()
                im_clean = im_bgr.copy()
                im_annotated = im_bgr.copy()
                H_img, W_img = im_annotated.shape[:2]
                t_labels = labels_out[labels_out[:, 0] == t_idx]
                for lbl in t_labels:
                    cls_id = int(lbl[1].item())
                    cx, cy, bw, bh = lbl[2:6].numpy()
                    px, py = int(cx * W_img), int(cy * H_img)
                    pw, ph = int(bw * W_img), int(bh * H_img)
                    x1, y1 = px - pw // 2, py - ph // 2
                    x2, y2 = px + pw // 2, py + ph // 2

                    if cls_id == 1:
                        cv2.rectangle(im_annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        if self.ft_coef > 0 and len(lbl) >= 6 + self.ft_length:
                            coefs = lbl[6 : 6 + self.ft_length].numpy().copy()
                            coefs[0] *= W_img; coefs[1] *= H_img
                            for k in range(self.ft_coef):
                                idx = 2 + k * 4
                                coefs[idx] *= W_img; coefs[idx+1] *= W_img
                                coefs[idx+2] *= H_img; coefs[idx+3] *= H_img
                            num_points = 36
                            t_vals = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                            x_curve = np.full_like(t_vals, coefs[0])
                            y_curve = np.full_like(t_vals, coefs[1])
                            for k in range(1, self.ft_coef + 1):
                                idx = 2 + (k - 1) * 4
                                ak, bk, ck, dk = coefs[idx:idx+4]
                                x_curve += ak * np.cos(k * t_vals) + bk * np.sin(k * t_vals)
                                y_curve += ck * np.cos(k * t_vals) + dk * np.sin(k * t_vals)
                            contour = np.vstack((x_curve, y_curve)).T.astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(im_annotated, [contour], isClosed=True, color=(255, 0, 255), thickness=1)
                    else:
                        cv2.rectangle(im_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(im_annotated, (px, py), 2, (0, 255, 0), -1)

                combined_img = np.concatenate((im_clean, im_annotated), axis=1)
                cv2.putText(combined_img, f"Original (Frame {t_idx})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_img, f"Clean Annotated (Frame {t_idx})", (W_img + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(debug_dir, f"frame_{t_idx}.jpg"), combined_img)

        return torch.from_numpy(final_imgs), labels_out, path_seq, (img_seq[0].shape[0], img_seq[0].shape[1])

    @staticmethod
    def collate_fn(batch):

        img, label, path_seqs, shapes = zip(*batch)
        seq_len = img[0].shape[0]


        for i, l in enumerate(label):
            if l.shape[0] > 0:
                t = l[:, 0].clone()
                l[:, 0] = i * seq_len + t

        if len(label) > 0: labels_out = torch.cat(label, 0)
        else: labels_out = torch.zeros((0, batch[0][1].shape[1]))

        imgs_out = torch.cat(img, 0)


        log_file = "debug_network_input.txt"
        import os

        if not os.path.exists(log_file) and len(labels_out) > 0:
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("="*85 + "\n")
                    f.write(" [NETWORK INPUT LOG]  GPU  Batch \n\n")
                    f.write(f"  (imgs_out): {imgs_out.shape} -> [BatchSize * SeqLen, Channels, Height, Width]\n")
                    f.write(f"  (labels_out): {labels_out.shape} -> [Batch, 1() + 1() + 4(xywh) + ...]\n\n")

                    f.write("  10  ( 0  19 ):\n")
                    f.write(str(labels_out[:10]) + "\n\n")

                    f.write("  Batch  ():\n")
                    f.write(str(labels_out[:, 0].unique().tolist()) + "\n")
                    f.write("="*85 + "\n")
            except Exception:
                pass


        paths_out = []
        shapes_out = []
        for p_seq, s in zip(path_seqs, shapes):
            paths_out.extend(p_seq)
            shapes_out.extend([s] * seq_len)


        return imgs_out, labels_out, tuple(paths_out), tuple(shapes_out)


