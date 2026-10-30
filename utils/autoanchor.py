
"""
Auto-anchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr, pts2dir
import os.path as osp


def check_anchor_order(m):

    a = m.anchors.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=[640,640]):

    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')

    ratio = np.minimum(imgsz[-1] / dataset.shapes[:, 0], imgsz[-2] / dataset.shapes[:, 1])
    shapes = ratio.reshape(-1, 1) * dataset.shapes


    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))

    m = model.module.get_module_byname('Detect') if hasattr(model, 'module') else model.get_module_byname('Detect')
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()

    def metric(wh,k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1 / thr).float().sum(1).mean()
        bpr = (best > 1 / thr).float().mean()
        return bpr, aat

    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)
    bpr, aat = metric(wh,anchors.cpu().view(-1, 2))
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:
        print('. Attempting to improve anchors, please wait...')
        na = m.anchors.numel() // 2
        try:
            if len(dataset.labels)>=na:
                anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
            else:
                print(f'\033[91min check_anchors->len(dataset.labels)={len(dataset.labels)} vs na={na}\033[0m')
                exit(0)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(wh,anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            check_anchor_order(m)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)
            m_ft = model.module.get_module_byname('FTInfer') if hasattr(model, 'module') else model.get_module_byname('FTInfer')
            if m_ft is not None:
                m_ft.anchors = m.anchors
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')


    m2 = model.module.get_module_byname('Detect2') if hasattr(model, 'module') else model.get_module_byname('Detect2')
    if m2 is not None:
        pts = torch.tensor(np.concatenate([l[:, 5:13] * s.reshape(1,2).repeat(4) for s, l in zip(shapes * scale, dataset.labels) if l.shape[0]>0])).float()

        ab = 2 * pts2dir(pts,1)[:,2:]

        anchors = m2.anchors.clone() * m2.stride.to(m2.anchors.device).view(-1, 1, 1)
        bpr, aat = metric(ab,anchors.cpu().view(-1, 2))
        print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
        if bpr < 0.98:
            print('. Attempting to improve anchors, please wait...')
            na = m2.anchors.numel() // 2
            try:
                anchors = kmean_anchors_ab(ab.numpy(), n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
            except Exception as e:
                print(f'{prefix}ERROR: {e}')
            new_bpr = metric(ab,anchors)[0]
            if new_bpr > bpr:
                anchors = torch.tensor(anchors, device=m2.anchors.device).type_as(m2.anchors)
                check_anchor_order(m2)
                m2.anchors[:] = anchors.clone().view_as(m2.anchors) / m2.stride.to(m2.anchors.device).view(-1, 1, 1)
                print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
            else:
                print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
        print('')

def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=[640,640], thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1 / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]

        return x, x.max(1)[0]

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

    def print_results(k):
        k = k[np.argsort(k.prod(1))]
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%.4f,%.4f' % (x[0], x[1]), end=',  ' if i < len(k) - 1 else '\n')
        return k

    if isinstance(dataset, str):
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        from utils.datasets import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(osp.join(data_dict['path'], data_dict['train']), augment=True, rect=True)


    ratio = np.minimum(img_size[-1] / dataset.shapes[:, 0], img_size[-2] / dataset.shapes[:, 1])
    shapes = ratio.reshape(-1, 1) * dataset.shapes

    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])


    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]


    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)
    k, dist = kmeans(wh / s, n, iter=30)
    assert len(k) == n, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k)


    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)


def kmean_anchors_ab(ab_objs, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1 / thr
    prefix = colorstr('autoanchor: ')

    def metric(ab_objs, k):
        r = ab_objs[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]

        return x, x.max(1)[0]

    def anchor_fitness(ab_objs, k):
        _, best = metric(ab_objs, torch.tensor(k, dtype=torch.float32))
        return (best * (best > thr).float()).mean()

    def print_results(ab_objs,k):
        k = k[np.argsort(k.prod(1))]
        x, best = metric(ab_objs, k)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')
        return k

    '''
    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(osp.join(data_dict['path'], data_dict['train']), augment=False, rect=False)

    # Get label ab_objs

    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    dir_labels = []
    for dir_l in dataset.labels:
        ls = [x[5:][None] for x in dir_l]
        if len(ls) == 0:
            continue
        tmp = np.concatenate(tuple(ls), axis=0)
        dir_labels.append(tmp)
    ab_objs = np.concatenate([l[:, 2:4] * s for s, l in zip(shapes, dir_labels)])  # wh
    '''

    i = (ab_objs < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(ab_objs)} labels are < 3 pixels in size.')
    ab = ab_objs[(ab_objs >= 2.0).any(1)]


    print(f'{prefix}Running kmeans for {n} anchors on {len(ab)} points...')
    s = ab.std(0)
    k, dist = kmeans(ab / s, n, iter=30)
    assert len(k) == n, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    ab = torch.tensor(ab, dtype=torch.float32)
    ab_objs = torch.tensor(ab_objs, dtype=torch.float32)
    k = print_results(ab_objs,k)


    npr = np.random
    f, sh, mp, s = anchor_fitness(ab_objs, k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(ab_objs, kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(ab_objs,k)

    return print_results(ab_objs,k)


