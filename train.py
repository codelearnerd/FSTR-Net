"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data data/infrared_e2.yaml --cfg models/repvit_m0_9_t2.yaml --img 640
"""


import cv2
from LSTM_video_loader import VideoFourierDataset

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import val
from models.experimental import attempt_load
from models.common import GLOBAL_ORACLE_PERIPHERAL_DICT
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, get_latest_run_exp, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer,
                           get_ft_num)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (EarlyStopping, ModelEMA, create_cuda_grad_scaler,
                               de_parallel, select_device, torch_distributed_zero_first)
from tools.plotbox import plot_one_rot_box

from datetime import datetime, timedelta

import shutil

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,
          opt,
          device,
          callbacks
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, fold =\
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg,\
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.fold


    historical_t2_validation = (
        Path(str(cfg)).name in {
            'repvit_m0_9_t2.yaml'})
    if historical_t2_validation and RANK in {-1, 0}:
        LOGGER.info(
            'Historical T2 compatibility: end-of-epoch validation uses the '
            'legacy first-order Fourier surrogate; final best_mAP50 report '
            'uses the complete reconstructed polygon.')


    seq_len = 5


    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / 'last.pt', w / 'best.pt'
    best_ap = best.with_name('best_mAP50.pt')


    p2f_train_stage = os.getenv('P2F_TRAIN_STAGE', '').strip().lower()


    hyp_file = save_dir / 'hyp.yaml'
    if(os.path.exists(hyp_file)):
        with open(hyp_file, errors='ignore') as f:
            hyp = yaml.safe_load(f)
    else:
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)
            LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))


        expand_demo = os.getenv('P2F_EXPAND_DEMO', '').strip().lower()
        expand_profiles = {
            'gt': {
                'ft_infer_coef': 0.001,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_cover_gt': 1.0,
                'ft_cover_margin': 1.5,
                'ft_cover_padding': 0.015625,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_pred': 0.0,
            },


            'gt_strong': {
                'ft_infer_coef': 0.00025,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_cover_gt': 4.0,
                'ft_cover_margin': 1.5,
                'ft_cover_padding': 0.015625,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_pred': 0.0,
            },


            'gt_refine': {
                'ft_infer_coef': 0.00025,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_cover_gt': 4.0,
                'ft_cover_margin': 1.5,
                'ft_cover_padding': 0.015625,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.05,
                'ft_cover_band': 0.003125,
                'ft_cover_pred': 0.0,
            },


            'shape_refine': {
                'ft_infer_coef': 0.00025,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.04,
                'ft_harmonic_shape': 0.10,
                'ft_cover_gt': 4.0,
                'ft_cover_margin': 1.5,
                'ft_cover_padding': 0.015625,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_pred': 0.0,
            },


            'directional_refine': {
                'ft_infer_coef': 0.00025,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.02,
                'ft_harmonic_shape': 0.05,
                'ft_directional': 1.0,
                'ft_directional_directions': 64,
                'ft_directional_under': 2.0,
                'ft_directional_over': 0.5,
                'ft_directional_band': 0.00625,
                'ft_curve_warmup_epochs': 2,
                'ft_cover_gt': 2.0,
                'ft_cover_margin': 1.0,
                'ft_cover_padding': 0.003125,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_pred': 0.0,
            },


            'directional_safe': {
                'ft_infer_coef': 0.00025,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.02,
                'ft_harmonic_shape': 0.05,
                'ft_directional': 1.0,
                'ft_directional_directions': 64,
                'ft_directional_under': 2.0,
                'ft_directional_over': 0.1,
                'ft_directional_band': 0.00625,
                'ft_curve_warmup_epochs': 2,
                'ft_cover_gt': 4.0,
                'ft_cover_margin': 1.0,
                'ft_cover_padding': 0.003125,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_hard_fraction': 0.35,
                'ft_cover_hard_gain': 2.0,
                'ft_cover_pred': 0.0,
            },
            'pred': {
                'ft_infer_coef': 0.001,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.5,
                'ft_cover_pred_threshold': 0.03,
                'ft_cover_pred_under_gain': 5.0,
                'ft_cover_pred_warmup_epochs': 1,
            },
        }
        if expand_demo:
            if expand_demo not in expand_profiles:
                raise ValueError(
                    f'Unknown P2F_EXPAND_DEMO={expand_demo!r}; choose gt, gt_strong, gt_refine, shape_refine, directional_refine, directional_safe, or pred')
            hyp.update(expand_profiles[expand_demo])
            LOGGER.warning(
                f'P2F expansion stress test={expand_demo}: '
                f'{expand_profiles[expand_demo]}')


        p2f_hyp_env = {
            'P2F_LR0': 'lr0',
            'P2F_LRF': 'lrf',
            'P2F_FT_COEF': 'ft_infer_coef',
            'P2F_FT_CURVE': 'ft_curve',
            'P2F_FT_AREA': 'ft_area',
            'P2F_FT_SHAPE': 'ft_shape',
            'P2F_FT_HARMONIC_SHAPE': 'ft_harmonic_shape',
            'P2F_FT_DIRECTIONAL': 'ft_directional',
            'P2F_FT_DIRECTIONAL_DIRECTIONS': 'ft_directional_directions',
            'P2F_FT_DIRECTIONAL_UNDER': 'ft_directional_under',
            'P2F_FT_DIRECTIONAL_OVER': 'ft_directional_over',
            'P2F_FT_DIRECTIONAL_BAND': 'ft_directional_band',
            'P2F_COVER_GT': 'ft_cover_gt',
            'P2F_COVER_MARGIN': 'ft_cover_margin',
            'P2F_COVER_PADDING': 'ft_cover_padding',
            'P2F_COVER_TOLERANCE': 'ft_cover_tolerance',
            'P2F_COVER_OVER_GAIN': 'ft_cover_over_gain',
            'P2F_COVER_BAND': 'ft_cover_band',
            'P2F_COVER_HARD_FRACTION': 'ft_cover_hard_fraction',
            'P2F_COVER_HARD_GAIN': 'ft_cover_hard_gain',
            'P2F_COVER_PRED': 'ft_cover_pred',
            'P2F_COVER_PRED_THRESHOLD': 'ft_cover_pred_threshold',
            'P2F_COVER_PRED_GAIN': 'ft_cover_pred_under_gain',
            'P2F_COVER_PRED_BOUNDARY': 'ft_cover_pred_boundary_gain',
        }
        applied_overrides = {}
        for env_name, hyp_name in p2f_hyp_env.items():
            value = os.getenv(env_name, '').strip()
            if value:
                hyp[hyp_name] = float(value)
                applied_overrides[hyp_name] = hyp[hyp_name]
        if applied_overrides:
            LOGGER.warning(f'P2F hyperparameter overrides: {applied_overrides}')


        valid_train_stages = {
            '', 'yolo_base', 'detect_temporal', 'point_only', 'f2',
            'feature_bridge', 'feature_joint', 'point_fourier_only',
            'dual_temporal_only', 'point_to_fourier_outward',
             'point_to_fourier_shape_refine', 'point_denoise',
             'point_to_fourier_stable', 'point_to_fourier_peripheral',
             'point_to_fourier_joint', 'fourier_candidate_backbone',
             'fourier_contour_expand', 'fourier_directional_geometry',
             'fourier_directional_oracle', 'fourier_point_rescue',
             'fourier_hybrid_match',
             'point_motion_gate',
             'point_motion_refine'}
        if p2f_train_stage not in valid_train_stages:
            raise ValueError(
                f'Unknown P2F_TRAIN_STAGE={p2f_train_stage!r}; '
                f'choose one of {tuple(sorted(valid_train_stages))}')
        if p2f_train_stage == 'yolo_base':
            yolo_only_hyp = {
                'ft_infer_xy': 0.0,
                'ft_infer_coef': 0.0,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0,
                'ft_directional': 0.0,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'p2f_train_stage': 'yolo_base',
            }
            hyp.update(yolo_only_hyp)
            LOGGER.warning(
                'P2F training stage=yolo_base: only YOLO box/object/class '
                'objectives can update the model')
        elif p2f_train_stage == 'detect_temporal':


            hyp.update({
                'ft_infer_xy': 0.0, 'ft_infer_coef': 0.0,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0, 'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0, 'biopoint_center': 0.0,
                'biopoint_offset': 0.0, 'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'p2f_train_stage': 'detect_temporal',
            })
            LOGGER.warning(
                'P2F training stage=detect_temporal: train Detect prediction '
                'convolutions + P3/P4/P5 causal adapters with YOLO losses only')
        elif p2f_train_stage == 'point_only':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.0, 'ft_infer_coef': 0.0,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0, 'formation_support': 0.0,
                'mask_loss': 0.0, 'p2f_train_stage': 'point_only',
            })
            LOGGER.warning(
                'P2F training stage=point_only: only the composite point '
                'objective is active; Detect/backbone/Fourier are frozen')
        elif p2f_train_stage == 'point_denoise':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.0, 'ft_infer_coef': 0.0,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0, 'mask_loss': 0.0,
                'biopoint_use_macro_gt': 1,
                'biopoint_sigma': 0.45,
                'biopoint_heatmap': 0.05,
                'biopoint_center': 0.30,
                'biopoint_offset': 0.025,
                'biopoint_wh': 0.025,
                'biopoint_region': 0.10,
                'biopoint_pos_gain': 40.0,
                'biopoint_center_pos_gain': 400.0,
                'biopoint_center_neg_gain': 0.50,
                'biopoint_focal_gamma': 2.0,
                'biopoint_outside_weight': 1.0,
                'p2f_train_stage': 'point_denoise',
            })
            LOGGER.warning(
                'P2F training stage=point_denoise: train BioCountInfer only; '
                'macro GT weights background suppression during training, '
                'while Fourier/proposal/backbone remain frozen')
        elif p2f_train_stage == 'point_motion_gate':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.0, 'ft_infer_coef': 0.0,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0, 'mask_loss': 0.0,
                'biocount_warmup_epochs': 3,
                'biopoint_use_macro_gt': 1,
                'biopoint_sigma': 0.45,
                'biopoint_heatmap': 0.05,
                'biopoint_center': 0.30,
                'biopoint_offset': 0.025,
                'biopoint_wh': 0.025,
                'biopoint_region': 0.10,
                'biopoint_pos_gain': 40.0,
                'biopoint_center_pos_gain': 400.0,
                'biopoint_center_neg_gain': 0.50,
                'biopoint_focal_gamma': 2.0,
                'biopoint_outside_weight': 1.0,
                'biopoint_motion_gate': 1.0,


                'biopoint_motion_pos_gain': 80.0,
                'biopoint_motion_focal_gamma': 2.0,
                'biopoint_motion_frame_weights': [
                    0.0, 0.25, 0.50, 0.75, 1.0],
                'p2f_train_stage': 'point_motion_gate',
            })
            LOGGER.warning(
                'P2F training stage=point_motion_gate: freeze the spatial '
                'point detector and every macro/Fourier path; train only the '
                'explicit motion gate with direct temporal heatmap supervision')
        elif p2f_train_stage == 'point_motion_refine':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.0001')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.0, 'ft_infer_coef': 0.0,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0, 'mask_loss': 0.0,
                'biocount_warmup_epochs': 1,
                'biopoint_use_macro_gt': 1,
                'biopoint_sigma': 0.45,
                'biopoint_heatmap': 0.05,
                'biopoint_center': 0.30,

                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.10,
                'biopoint_pos_gain': 40.0,
                'biopoint_center_pos_gain': 400.0,
                'biopoint_center_neg_gain': 1.0,
                'biopoint_focal_gamma': 2.0,
                'biopoint_outside_weight': 1.0,
                'biopoint_motion_gate': 1.0,
                'biopoint_motion_pos_gain': 80.0,
                'biopoint_motion_focal_gamma': 2.0,
                'biopoint_motion_frame_weights': [
                    0.0, 0.25, 0.50, 0.75, 1.0],


                'biopoint_hard_peak': float(os.getenv(
                    'P2F_HARD_PEAK_WEIGHT', '0.25')),
                'biopoint_hard_peak_topk': int(os.getenv(
                    'P2F_HARD_PEAK_TOPK', '100')),
                'biopoint_hard_peak_exclusion_radius': int(os.getenv(
                    'P2F_HARD_PEAK_RADIUS', '3')),
                'biopoint_hard_peak_threshold': float(os.getenv(
                    'P2F_HARD_PEAK_THRESHOLD', '0.30')),
                'biopoint_hard_peak_warmup_epochs': int(os.getenv(
                    'P2F_HARD_PEAK_WARMUP', '2')),
                'p2f_train_stage': 'point_motion_refine',
            })
            LOGGER.warning(
                'P2F training stage=point_motion_refine: freeze backbone, '
                'Fourier, point decoder and regression heads; jointly train '
                'the explicit temporal adapter and detached point cls_head')
        elif p2f_train_stage == 'point_fourier_only':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 1.0,
                'ft_infer_xy': 0.1, 'ft_infer_coef': 0.00625,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0, 'formation_support': 0.0,
                'mask_loss': 0.0,
                'p2f_train_stage': 'point_fourier_only',
            })
            LOGGER.warning(
                'P2F training stage=point_fourier_only: no YOLO Detect head; '
                'active objectives are Fourier proposal objectness, ft_xy, '
                'ft_coef and the detached point-response objective')
        elif p2f_train_stage == 'dual_temporal_only':
            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.1, 'ft_infer_coef': 0.00625,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0, 'formation_support': 0.0,
                'mask_loss': 0.0,
                'p2f_train_stage': 'dual_temporal_only',
            })
            LOGGER.warning(
                'P2F training stage=dual_temporal_only: freeze all spatial '
                'parameters and train only Point + P3/P4/P5 Fourier causal '
                'temporal adapters')
        elif p2f_train_stage == 'point_to_fourier_outward':
            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.1, 'ft_infer_coef': 0.00625,
                'ft_curve': 0.0, 'ft_area': 0.0, 'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0, 'ft_directional': 0.0,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 1.0,
                'ft_cover_pred_threshold': 0.12,
                'ft_cover_pred_under_gain': 4.0,
                'ft_cover_pred_directions': 64,
                'ft_cover_pred_warmup_epochs': 3,
                'ft_support_consistency': 0.0, 'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0, 'biopoint_center': 0.0,
                'biopoint_offset': 0.0, 'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'p2f_train_stage': 'point_to_fourier_outward',
            })
            LOGGER.warning(
                'P2F training stage=point_to_fourier_outward: Point is a '
                'frozen detached teacher; train outward bridges + FTInfer '
                'with native Fourier losses and outward-only predicted-point '
                'coverage')
        elif p2f_train_stage == 'point_to_fourier_shape_refine':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 1.0,
                'ft_infer_xy': 0.1, 'ft_infer_coef': 0.00625,
                'ft_curve': 0.0, 'ft_area': 0.0,
                'ft_shape': 0.05,
                'ft_harmonic_shape': 0.10,
                'ft_directional': 0.50,
                'ft_directional_directions': 64,
                'ft_directional_under': 2.0,
                'ft_directional_over': 0.50,
                'ft_directional_band': 0.003125,
                'ft_curve_warmup_epochs': 3,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.35,
                'ft_cover_pred_threshold': 0.20,
                'ft_cover_pred_under_gain': 2.0,
                'ft_cover_pred_boundary_gain': 0.20,
                'ft_cover_pred_directions': 64,
                'ft_cover_pred_warmup_epochs': 3,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0, 'biopoint_center': 0.0,
                'biopoint_offset': 0.0, 'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'p2f_train_stage': 'point_to_fourier_shape_refine',
            })
            LOGGER.warning(
                'P2F training stage=point_to_fourier_shape_refine: frozen '
                'Point/backbone teacher; train FourierProposal, three outward '
                'bridges and FTInfer with weak point coverage plus normalized '
                'harmonic/full-angle shape constraints')
        elif p2f_train_stage == 'point_to_fourier_stable':


            hyp.update({
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.1, 'ft_infer_coef': 0.0125,
                'ft_curve': 0.0, 'ft_area': 0.0,
                'ft_shape': 0.005,
                'ft_harmonic_shape': 0.01,
                'ft_directional': 0.02,
                'ft_directional_directions': 64,
                'ft_directional_under': 1.0,
                'ft_directional_over': 1.0,
                'ft_directional_band': 0.003125,
                'ft_curve_warmup_epochs': 3,
                'ft_cover_gt': 0.0, 'ft_cover_pred': 0.0,
                'ft_cover_pred_boundary_gain': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0, 'mask_loss': 0.0,
                'biopoint_heatmap': 0.0, 'biopoint_center': 0.0,
                'biopoint_offset': 0.0, 'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'p2f_train_stage': 'point_to_fourier_stable',
            })
            LOGGER.warning(
                'P2F training stage=point_to_fourier_stable: frozen denoised '
                'Point/proposal/backbone; train low-gain bridges + FTInfer '
                'with anchored native coefficients and weak shape constraints')
        elif p2f_train_stage == 'point_to_fourier_peripheral':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.0002')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'ft_infer_xy': 0.10,
                'ft_infer_coef': 0.0125,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.01,
                'ft_harmonic_shape': 0.02,
                'ft_directional': 0.05,
                'ft_directional_directions': 64,
                'ft_directional_under': 1.25,
                'ft_directional_over': 0.75,
                'ft_directional_band': 0.003125,
                'ft_curve_warmup_epochs': 3,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.0,
                'ft_cover_pred_boundary_gain': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'point_to_fourier_peripheral',
            })
            LOGGER.warning(
                'P2F training stage=point_to_fourier_peripheral: freeze '
                'cleaned Point/FourierProposal/backbone; train three detached '
                'peripheral bridges + FTInfer/FT temporal adapters from native '
                '.ft center, coefficient and weak shape supervision')
        elif p2f_train_stage == 'fourier_contour_expand':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.00005')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0,
                'cls': 0.0,
                'obj': 0.0,


                'anchor_t': float(
                    os.getenv('P2F_CANDIDATE_ANCHOR_T', '4.0')),
                'ft_infer_xy': float(os.getenv('P2F_FT_XY', '0.20')),
                'ft_infer_coef': float(
                    os.getenv('P2F_FT_COEF', '0.00625')),
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': float(os.getenv('P2F_FT_SHAPE', '0.02')),
                'ft_harmonic_shape': float(
                    os.getenv('P2F_FT_HARMONIC_SHAPE', '0.04')),
                'ft_directional': float(
                    os.getenv('P2F_FT_DIRECTIONAL', '0.15')),
                'ft_directional_directions': 64,
                'ft_directional_under': float(
                    os.getenv('P2F_FT_DIRECTIONAL_UNDER', '2.0')),
                'ft_directional_over': float(
                    os.getenv('P2F_FT_DIRECTIONAL_OVER', '0.5')),
                'ft_directional_band': float(
                    os.getenv('P2F_FT_DIRECTIONAL_BAND', '0.00625')),
                'ft_curve_warmup_epochs': 3,
                'ft_cover_gt': float(os.getenv('P2F_COVER_GT', '1.0')),
                'ft_cover_margin': float(
                    os.getenv('P2F_COVER_MARGIN', '1.0')),
                'ft_cover_padding': float(
                    os.getenv('P2F_COVER_PADDING', '0.003125')),
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_hard_fraction': float(
                    os.getenv('P2F_COVER_HARD_FRACTION', '0.35')),
                'ft_cover_hard_gain': float(
                    os.getenv('P2F_COVER_HARD_GAIN', '2.0')),
                'ft_cover_pred': float(
                    os.getenv('P2F_COVER_PRED', '0.30')),
                'ft_cover_pred_threshold': float(
                    os.getenv('P2F_COVER_PRED_THRESHOLD', '0.40')),
                'ft_cover_pred_under_gain': float(
                    os.getenv('P2F_COVER_PRED_GAIN', '3.0')),
                'ft_cover_pred_boundary_gain': float(
                    os.getenv('P2F_COVER_PRED_BOUNDARY', '0.05')),
                'ft_cover_pred_directions': 64,
                'ft_cover_pred_warmup_epochs': 3,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'fourier_contour_expand',
            })
            LOGGER.warning(
                'P2F training stage=fourier_contour_expand: freeze calibrated '
                'candidate backbone/Proposal and Point teacher; train only '
                'scale-specific bridges + FTInfer/temporal adapters with GT '
                'lower-envelope, predicted-point coverage and native FT '
                'directional shape constraints')
        elif p2f_train_stage == 'fourier_directional_geometry':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.00005')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0,
                'cls': 0.0,
                'obj': 0.0,
                'anchor_t': float(
                    os.getenv('P2F_CANDIDATE_ANCHOR_T', '4.0')),
                'ft_infer_xy': float(os.getenv('P2F_FT_XY', '0.20')),
                'ft_infer_coef': float(
                    os.getenv('P2F_FT_COEF', '0.0125')),
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': float(os.getenv('P2F_FT_SHAPE', '0.03')),
                'ft_harmonic_shape': float(
                    os.getenv('P2F_FT_HARMONIC_SHAPE', '0.06')),
                'ft_directional': float(
                    os.getenv('P2F_FT_DIRECTIONAL', '0.25')),
                'ft_directional_directions': 64,
                'ft_directional_under': float(
                    os.getenv('P2F_FT_DIRECTIONAL_UNDER', '1.5')),
                'ft_directional_over': float(
                    os.getenv('P2F_FT_DIRECTIONAL_OVER', '1.0')),
                'ft_directional_band': float(
                    os.getenv('P2F_FT_DIRECTIONAL_BAND', '0.003125')),
                'ft_curve_warmup_epochs': 2,
                'ft_cover_gt': float(os.getenv('P2F_COVER_GT', '0.40')),
                'ft_cover_margin': float(
                    os.getenv('P2F_COVER_MARGIN', '1.0')),
                'ft_cover_padding': float(
                    os.getenv('P2F_COVER_PADDING', '0.003125')),
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_hard_fraction': 0.35,
                'ft_cover_hard_gain': 1.5,


                'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'fourier_directional_geometry',
            })
            LOGGER.warning(
                'P2F training stage=fourier_directional_geometry: freeze '
                'candidate/Point/backbone teachers; train three '
                'CandidateDirectionalFourierBridge modules and FTInfer using '
                'candidate-gated directional support, fixed DFT projection, '
                'native coefficients and asymmetric shape constraints')
        elif p2f_train_stage == 'fourier_directional_oracle':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.00005')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'anchor_t': float(
                    os.getenv('P2F_CANDIDATE_ANCHOR_T', '4.0')),

                'ft_infer_xy': 0.0,
                'ft_infer_coef': float(
                    os.getenv('P2F_FT_COEF', '0.0125')),
                'ft_curve': 0.0, 'ft_area': 0.0,
                'ft_shape': float(os.getenv('P2F_FT_SHAPE', '0.03')),
                'ft_harmonic_shape': float(
                    os.getenv('P2F_FT_HARMONIC_SHAPE', '0.06')),
                'ft_directional': float(
                    os.getenv('P2F_FT_DIRECTIONAL', '0.25')),
                'ft_directional_directions': 64,
                'ft_directional_under': 1.5,
                'ft_directional_over': 1.0,
                'ft_directional_band': 0.003125,
                'ft_curve_warmup_epochs': 2,
                'ft_cover_gt': float(os.getenv('P2F_COVER_GT', '0.40')),
                'ft_cover_margin': 1.0,
                'ft_cover_padding': 0.003125,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': 0.0,
                'ft_cover_band': 0.0,
                'ft_cover_hard_fraction': 0.35,
                'ft_cover_hard_gain': 1.5,
                'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'fourier_directional_oracle',
            })
            LOGGER.warning(
                'P2F training stage=fourier_directional_oracle: macro GT '
                'controls one candidate and its center; micro GT supplies '
                'directional support; only FTInfer is optimized against native '
                'Fourier coefficient/shape targets')
        elif p2f_train_stage in {
                'fourier_point_rescue', 'fourier_hybrid_match'}:


            robust_hybrid = p2f_train_stage == 'fourier_hybrid_match'
            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.00005')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0, 'cls': 0.0, 'obj': 0.0,
                'anchor_t': float(
                    os.getenv('P2F_CANDIDATE_ANCHOR_T', '4.0')),
                'ft_infer_xy': float(os.getenv('P2F_FT_XY', '0.20')),
                'ft_infer_coef': float(
                    os.getenv('P2F_FT_COEF', '0.0125')),
                'ft_curve': 0.0, 'ft_area': 0.0,
                'ft_shape': float(os.getenv('P2F_FT_SHAPE', '0.03')),
                'ft_harmonic_shape': float(
                    os.getenv('P2F_FT_HARMONIC_SHAPE', '0.06')),
                'ft_directional': float(
                    os.getenv('P2F_FT_DIRECTIONAL', '0.25')),
                'ft_directional_directions': 64,
                'ft_directional_under': float(
                    os.getenv('P2F_FT_DIRECTIONAL_UNDER', '1.5')),
                'ft_directional_over': float(
                    os.getenv(
                        'P2F_FT_DIRECTIONAL_OVER',
                        '1.20' if robust_hybrid else '1.0')),
                'ft_directional_band': float(
                    os.getenv('P2F_FT_DIRECTIONAL_BAND', '0.003125')),
                'ft_curve_warmup_epochs': 2,
                'ft_cover_gt': float(os.getenv('P2F_COVER_GT', '0.40')),
                'ft_cover_margin': 1.0,
                'ft_cover_padding': 0.003125,
                'ft_cover_tolerance': 0.0,
                'ft_cover_over_gain': float(os.getenv(
                    'P2F_COVER_OVER_GAIN',
                    '0.20' if robust_hybrid else '0.0')),
                'ft_cover_band': float(os.getenv(
                    'P2F_COVER_BAND',
                    '0.003125' if robust_hybrid else '0.0')),
                'ft_cover_hard_fraction': 0.35,
                'ft_cover_hard_gain': 1.5,


                'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': p2f_train_stage,
            })
            if p2f_train_stage == 'fourier_hybrid_match':
                LOGGER.warning(
                    'P2F training stage=fourier_hybrid_match: freeze calibrated '
                    'Point/FourierProposal/backbone teachers; Fourier scores '
                    'and point-set extent select one scale/anchor per cluster, '
                    'then train bridges + FTInfer with confidence/density '
                    'filtered candidate-local discrete peaks, one-candidate '
                    'point assignment, robust extent-center correction, '
                    'directional point envelope, angular spike trimming, '
                    'bounded outward-only expansion and a light over-coverage '
                    'penalty while retaining image-predicted base coefficients')
            else:
                LOGGER.warning(
                    'P2F training stage=fourier_point_rescue: freeze calibrated '
                    'Point/FourierProposal/backbone teachers; train three bridges '
                    '+ FTInfer with P3 point-set candidate rescue, predicted local '
                    'centroid correction and directional DFT reconstruction')
        elif p2f_train_stage == 'fourier_candidate_backbone':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.0001')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0,
                'cls': 0.0,
                'obj': float(os.getenv('P2F_CANDIDATE_OBJ', '1.0')),
                'anchor_t': float(os.getenv('P2F_CANDIDATE_ANCHOR_T', '4.0')),
                'ft_infer_xy': 0.0,
                'ft_infer_coef': 0.0,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.0,
                'ft_harmonic_shape': 0.0,
                'ft_directional': 0.0,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.0,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'fourier_candidate_backbone',
            })
            LOGGER.warning(
                'P2F training stage=fourier_candidate_backbone: keep point '
                'input layers 0..4 fixed; train deep shared backbone/neck, '
                'Fourier feature branch and FourierProposal with macro '
                'objectness; freeze Point, bridges and FTInfer')
        elif p2f_train_stage == 'point_to_fourier_joint':


            hyp.update({
                'lr0': float(os.getenv('P2F_LR0', '0.0001')),
                'lrf': float(os.getenv('P2F_LRF', '0.1')),
                'box': 0.0,
                'cls': 0.0,
                'obj': 1.0,
                'ft_infer_xy': 0.25,
                'ft_infer_coef': 0.0125,
                'ft_curve': 0.0,
                'ft_area': 0.0,
                'ft_shape': 0.01,
                'ft_harmonic_shape': 0.02,
                'ft_directional': 0.05,
                'ft_directional_directions': 64,
                'ft_directional_under': 1.25,
                'ft_directional_over': 0.75,
                'ft_directional_band': 0.003125,
                'ft_curve_warmup_epochs': 3,
                'ft_cover_gt': 0.0,
                'ft_cover_pred': 0.15,
                'ft_cover_pred_threshold': 0.55,
                'ft_cover_pred_under_gain': 2.0,
                'ft_cover_pred_boundary_gain': 0.10,
                'ft_cover_pred_directions': 64,
                'ft_cover_pred_warmup_epochs': 5,
                'ft_support_consistency': 0.0,
                'formation_support': 0.0,
                'mask_loss': 0.0,
                'biopoint_heatmap': 0.0,
                'biopoint_center': 0.0,
                'biopoint_offset': 0.0,
                'biopoint_wh': 0.0,
                'biopoint_region': 0.0,
                'biopoint_motion_gate': 0.0,
                'biopoint_hard_peak': 0.0,
                'p2f_train_stage': 'point_to_fourier_joint',
            })
            LOGGER.warning(
                'P2F training stage=point_to_fourier_joint: freeze cleaned '
                'Point/backbone; jointly train FourierProposal, scale-specific '
                'peripheral bridges and FTInfer/FT temporal adapters with '
                'objectness, native .ft and weak directional point coverage')

        with open(hyp_file, 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)


    from general.global_cfg import get_machine_info
    opt.machine_info = get_machine_info()
    opt.version = os.path.dirname(os.path.abspath(__file__))
    opt_dict = vars(opt)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(opt_dict, f, sort_keys=False)
    data_dict = None


    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp


        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))


    assert os.path.join(str(save_dir))
    if not resume:
        assert os.path.join(opt.cfg)
        shutil.copy2(opt.cfg, os.path.join(str(save_dir), os.path.basename(opt.cfg)))


    plots = not evolve if opt.plots else False
    cuda = device.type != 'cpu'


    experiment_seed = int(getattr(opt, 'experiment_seed', 0))
    init_seeds(experiment_seed + max(RANK, 0))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    assert(os.path.exists(data_dict['path']))

    train_path, val_path = data_dict['train'], data_dict['val']


    nc = 1 if single_cls else int(data_dict['nc'])
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'


    mask_dir = data_dict.get('mask_dir', 0)
    ft_coef = data_dict.get('ft_coef', 0)
    lamda = data_dict.get('lamda', 1.0)
    d0 = data_dict.get('d0', 2.0)
    if ft_coef == -1:
        ft_coef = get_ft_num(train_path)

    assert isinstance(mask_dir, (int, list)), f'{mask_dir}, Mask Dir Error.'
    mask_dir = [mask_dir] * nc if isinstance(mask_dir, int) else mask_dir
    if len(mask_dir) != nc:
        mask_dir = mask_dir + [0] * (nc - len(mask_dir))

    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')


    if resume:
        cfg = None

    d2_flag = False
    if cfg is not None:
        subclass_infer, value_infer = 0, 0
        last_hindex = -1
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg_dict = yaml.safe_load(f)
        ft_flag = False
        for i, h in enumerate(cfg_dict['head']):
            if h[2] in ['Detect', 'Detect2', 'SubclassInfer', 'ValueInfer', 'FTInfer']:
                if last_hindex == -1:
                    last_hindex = i
                if h[2] == 'Detect2':
                    d2_flag = True
                if h[2] == 'FTInfer':
                    ft_flag = True


    check_suffix(weights, ('.pt','.pth'))
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        if(os.path.basename(weights)=='last.pt'):
            if(ckpt['epoch']==-1):
                ckpt['epochs'] = opt.epochs
                ckpt['epoch'] = opt.epochs - 1
        model = Model(cfg or ckpt['model'].yaml, ch=data_dict.get('ch', 3), nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)


        source_anchors = ckpt['model'].yaml.get('anchors')
        target_anchors = model.yaml.get('anchors')
        anchors_overridden = bool(hyp.get('anchors'))
        anchors_changed = source_anchors != target_anchors
        exclude = (
            ['anchor']
            if not resume and (anchors_overridden or anchors_changed)
            else [])
        csd = ckpt['model'].float().state_dict()
        if (model.yaml.get('backbone_name') == 'repvit_m0_9' and
                ckpt['model'].yaml.get('backbone_name') != 'repvit_m0_9'):


            csd = {key: value for key, value in csd.items()
                   if not (key.startswith('model.') and
                           key.split('.')[1].isdigit() and
                           int(key.split('.')[1]) < 10)}
            LOGGER.info('RepViT-M0.9: excluded source backbone layers 0..9; '
                        'loading only compatible downstream checkpoint tensors')

        target_topology = (
            bool(model.yaml.get('point_heatmap_fourier_fusion', False)),
            bool(model.yaml.get('formation_support_bridge', False)),
            bool(model.yaml.get('point_fourier_feature_bridge', False)),
            bool(model.yaml.get('oracle_peripheral_gt', False)),
            any(h[2] == 'Detect' for h in model.yaml.get('head', [])),
        )
        source_topology = (
            bool(ckpt['model'].yaml.get('point_heatmap_fourier_fusion', False)),
            bool(ckpt['model'].yaml.get('formation_support_bridge', False)),
            bool(ckpt['model'].yaml.get('point_fourier_feature_bridge', False)),
            bool(ckpt['model'].yaml.get('oracle_peripheral_gt', False)),
            any(h[2] == 'Detect' for h in ckpt['model'].yaml.get('head', [])),
        )
        source_module_types = [
            type(module).__name__ for module in ckpt['model'].model]
        target_module_types = [
            type(module).__name__ for module in model.model]
        if (target_topology != source_topology and
                source_module_types != target_module_types):


            source_model = ckpt['model'].model
            target_model = model.model
            head_names = ('Detect', 'FTInfer', 'BioCountInfer')
            source_indices = {
                name: next((i for i, module in enumerate(source_model)
                            if type(module).__name__ == name), None)
                for name in head_names
            }
            target_indices = {
                name: next((i for i, module in enumerate(target_model)
                            if type(module).__name__ == name), None)
                for name in head_names
            }
            existing = [i for i in (*source_indices.values(), *target_indices.values())
                        if i is not None]
            if existing:
                first_changed = min(existing)
                csd = {
                    key: value for key, value in csd.items()
                    if not (key.startswith('model.') and
                            key.split('.')[1].isdigit() and
                            int(key.split('.')[1]) >= first_changed)
                }
                source_state = ckpt['model'].float().state_dict()
                for name in head_names:
                    source_idx, target_idx = source_indices[name], target_indices[name]
                    if source_idx is None or target_idx is None:
                        continue
                    source_prefix = f'model.{source_idx}.'
                    target_prefix = f'model.{target_idx}.'
                    for key, value in source_state.items():
                        if key.startswith(source_prefix):
                            csd[target_prefix + key[len(source_prefix):]] = value
                LOGGER.info(f'Remapped pretrained heads by type: {source_indices} -> {target_indices}')
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'\033[92mTransferred {len(csd)}/{len(model.state_dict())} items from {weights}\033[0m')
    else:
        model = Model(cfg, ch=data_dict.get('ch', 3), nc=nc, anchors=hyp.get('anchors'), ft_coef=ft_coef).to(device)


    if hasattr(model, 'set_seq_len'):
        model.set_seq_len(seq_len)


    point_backbone_grad = False
    point_gradient_head = de_parallel(model).get_module_byname(
        'BioCountInfer')
    if point_gradient_head is not None:
        point_backbone_grad = not bool(getattr(
            point_gradient_head, 'detach_features', True))
        LOGGER.warning(
            'Point -> shared-backbone gradient: '
            f"{'ENABLED' if point_backbone_grad else 'DETACHED'} "
            f'(BioCountInfer.detach_features='
            f'{not point_backbone_grad})')
        model_yaml = getattr(de_parallel(model), 'yaml', {})
        if ('point_backbone_grad' in model_yaml and
                bool(model_yaml['point_backbone_grad']) !=
                point_backbone_grad):
            raise ValueError(
                'Model YAML point_backbone_grad marker does not match '
                'BioCountInfer.detach_features')
        if bool(model_yaml.get('point_recall_recovery', False)):
            if not point_backbone_grad:
                raise ValueError(
                    'point_recall_recovery requires non-detached '
                    'BioCountInfer features')
            LOGGER.warning(
                'Point recall recovery: '
                f"output_floor={float(getattr(point_gradient_head, 'motion_output_floor', 0.0)):.3g}, "
                f"output_gamma={float(getattr(point_gradient_head, 'motion_output_gamma', 0.0)):.3g}, "
                f"heatmap={float(hyp.get('biopoint_heatmap', 0.0)):.3g}, "
                f"center={float(hyp.get('biopoint_center', 0.0)):.3g}, "
                f"motion={float(hyp.get('biopoint_motion_gate', 0.0)):.3g}, "
                f"pos_gain={float(hyp.get('biopoint_pos_gain', 0.0)):.3g}, "
                f"center_pos={float(hyp.get('biopoint_center_pos_gain', 0.0)):.3g}, "
                f"center_neg={float(hyp.get('biopoint_center_neg_gain', 0.0)):.3g}")
    point_temporal_align_override = os.getenv(
        'P2F_POINT_TEMPORAL_ALIGN', '').strip().lower()
    if point_temporal_align_override:
        if point_temporal_align_override not in {
                '0', '1', 'false', 'true', 'no', 'yes', 'off', 'on'}:
            raise ValueError(
                'P2F_POINT_TEMPORAL_ALIGN must be one of '
                '0/1, false/true, no/yes, or off/on')
        align_enabled = point_temporal_align_override in {
            '1', 'true', 'yes', 'on'}
        point_model = de_parallel(model)
        point_head = point_model.get_module_byname('BioCountInfer')
        point_adapter = getattr(point_head, 'temporal_adapter', None)
        if (point_adapter is None or
                point_adapter.__class__.__name__ !=
                'ExplicitMotionPointTemporalAdapter'):
            raise ValueError(
                'P2F_POINT_TEMPORAL_ALIGN requires an explicit '
                'BioCountInfer motion adapter')
        point_adapter.align = align_enabled
        point_model.yaml['point_temporal_align'] = align_enabled
        LOGGER.warning(
            'Point temporal alignment override: '
            f'align={align_enabled} (saved in checkpoint model YAML)')
    if p2f_train_stage in {'fourier_point_rescue', 'fourier_hybrid_match'}:
        candidate_bridges = [
            module for module in de_parallel(model).modules()
            if module.__class__.__name__ ==
            'CandidateDirectionalFourierBridge']
        if p2f_train_stage == 'fourier_point_rescue':


            for module in candidate_bridges:
                module.hybrid_match = False
                module.hybrid_builder = False
        else:
            hybrid_builders = [
                module for module in candidate_bridges
                if bool(getattr(module, 'hybrid_builder', False))]
            if len(candidate_bridges) != 3 or len(hybrid_builders) != 1:
                raise ValueError(
                    'fourier_hybrid_match requires three candidate bridges '
                    f'and exactly one arbitration builder; found '
                    f'{len(candidate_bridges)} and {len(hybrid_builders)}')
    if p2f_train_stage == 'fourier_directional_oracle':
        oracle_model = de_parallel(model)
        oracle_model.yaml['oracle_peripheral_gt'] = True
        proposal_count = bridge_count = ft_count = 0
        for module in oracle_model.modules():
            if module.__class__.__name__ == 'FourierProposal':
                module.oracle_geometry = True
                proposal_count += 1
            elif module.__class__.__name__ ==\
                    'CandidateDirectionalFourierBridge':


                module.max_gain = 0.0
                module.oracle_geometry = True
                bridge_count += 1
            elif module.__class__.__name__ == 'FTInfer':
                module.geometry_oracle = True
                ft_count += 1
        if proposal_count != 1 or bridge_count != 3 or ft_count != 1:
            raise ValueError(
                'fourier_directional_oracle requires one FourierProposal, '
                'three CandidateDirectionalFourierBridge modules and one '
                f'FTInfer; found {proposal_count}, {bridge_count}, {ft_count}')
        LOGGER.info(
            'Strict directional Oracle enabled: GT macro candidate/center + '
            'GT micro support; predicted point feature residual disabled')
    point_temporal_many_to_many = bool(
        getattr(de_parallel(model), 'yaml', {}).get(
            'point_temporal_many_to_many', False))
    oracle_switch = os.getenv('ORACLE_GT_ENABLED', '1').strip().lower()
    oracle_peripheral_gt = (
        bool(getattr(de_parallel(model), 'yaml', {}).get(
            'oracle_peripheral_gt', False)) and
        oracle_switch not in {'0', 'false', 'no', 'off'})
    if bool(getattr(de_parallel(model), 'yaml', {}).get(
            'oracle_peripheral_gt', False)):
        LOGGER.info(
            f'Oracle peripheral GT input enabled={oracle_peripheral_gt} '
            f'(ORACLE_GT_ENABLED={oracle_switch!r})')
    if point_temporal_many_to_many:
        LOGGER.info(
            'Point temporal training: causal many-to-many supervision is active; '
            'frames 1..T use prefixes [1], [1..2], ..., [1..T]')


    freeze = [f'model.{x}.' for x in range(freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False


    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)


    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz)


    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []
    grouped_param_ids = set()

    def add_param_once(group, param):
        """Keep optimizer groups disjoint while supporting custom scalar parameters."""
        if isinstance(param, nn.Parameter) and param.requires_grad and id(param) not in grouped_param_ids:
            group.append(param)
            grouped_param_ids.add(id(param))

    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            add_param_once(g2, v.bias)
        if isinstance(v, nn.BatchNorm2d):
            add_param_once(g0, v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            add_param_once(g1, v.weight)


    standalone_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in grouped_param_ids:
            add_param_once(g0, param)
            standalone_params.append(name)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': g2})
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} no decay, {len(g1)} weight decay, {len(g2)} bias")
    if standalone_params:
        LOGGER.info(f"{colorstr('optimizer:')} added {len(standalone_params)} standalone trainable parameter(s): "
                    + ', '.join(standalone_params))
    del g0, g1, g2


    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    ema = ModelEMA(model) if RANK in [-1, 0] else None


    start_epoch, best_fitness, best_ap50 = 0, 0.0, 0.0
    if pretrained:
        if resume:

            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']
                best_ap50 = ckpt.get('best_ap50', 0.0)


            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']


            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'

        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']

        del ckpt, csd


    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)


    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
    if resume:
        d2_flag = model.get_module_byname('Detect2') is not None

    pts = d2_flag


    LOGGER.info(f'\n?[Video Mode] Initializing VideoFourierDataset with seq_len={seq_len}...')


    dataset = VideoFourierDataset(
        path=train_path,
        img_size=imgsz,
        seq_length=seq_len,
        ft_coef=ft_coef,
        augment=True,
        hyp=hyp
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size // WORLD_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=VideoFourierDataset.collate_fn
    )


    nb = len(train_loader)
    mlc = nc - 1


    if RANK in [-1, 0]:
        val_loader = None
        if not noval:
            val_dataset = VideoFourierDataset(
                path=val_path,
                img_size=imgsz,
                seq_length=seq_len,
                ft_coef=ft_coef,
                augment=False,
                hyp=hyp
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size // WORLD_SIZE ,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                collate_fn=VideoFourierDataset.collate_fn
            )


    if not resume:
        labels = np.concatenate(dataset.labels, 0)


        if not opt.noautoanchor:

            pass

        model.half().float()

    callbacks.run('on_pretrain_routine_end')


    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


    m = de_parallel(model).get_module_byname('Detect')
    m = m if m is not None else de_parallel(model).get_module_byname('FourierProposal')
    if m is None:
        raise ValueError('Training requires Detect or FourierProposal')
    nl = m.nl
    hyp['box'] *= 3 / nl
    hyp['cls'] *= nc / 80 * 3 / nl
    hyp['obj'] *= ((imgsz if isinstance(imgsz,int) else max(imgsz)) / 640) ** 2 * 3. / nl


    mft = de_parallel(model).get_module_byname('FTInfer')
    model_mft = mft


    mft = ema.ema.get_module_byname('FTInfer')
    mft.cm = torch.from_numpy(dataset.cm).to(device=device)
    mft.ec = hyp.get('ec',0.02)


    model_mft.cm = mft.cm
    model_mft.ec = mft.ec
    LOGGER.info(
        f'FTInfer coefficient scales synchronized for train/EMA: '
        f'ec={float(mft.ec):g}, orders={mft.cm.numel()}')


    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp


    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc


    model.names = names
    model.mask_dir = mask_dir
    data_dict['mask_dir'] = deepcopy(model.mask_dir)


    t0 = time.time()
    warmup_min_iters = int(hyp.get('warmup_min_iters', 1000))
    nw = max(round(hyp['warmup_epochs'] * nb), warmup_min_iters)

    last_opt_step = -1
    maps = np.zeros(nc)
    results = [0 for i in range(10)]
    scheduler.last_epoch = start_epoch - 1


    scaler = create_cuda_grad_scaler(
        enabled=cuda, init_scale=1024.0,
        growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)


    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model,hungarian=hyp.get('hm',0), sort_obj_iou=hyp.get('sort_obj_iou',0),cm=mft.cm,ec=mft.ec)
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to \033[32m{colorstr('bold', save_dir)}\033[0m\n"
                f'Starting training for {epochs} epochs...')
    last_save_time = time.time()
    epoch = start_epoch
    mloss = None


    pcbi_bridge_only_epochs = max(
        int(os.getenv('PCBI_BRIDGE_ONLY_EPOCHS', '0')), 0)
    pcbi_bridge_param_ids = set()
    if pcbi_bridge_only_epochs > 0:
        pcbi_model = de_parallel(model)
        for module in pcbi_model.modules():
            if module.__class__.__name__ == 'PointFeatureGate':
                pcbi_bridge_param_ids.update(id(p) for p in module.parameters())
        point_head = pcbi_model.get_module_byname('BioCountInfer')
        if point_head is not None:
            for attr in ('macro_proj', 'macro_proj_extra'):
                adapter = getattr(point_head, attr, None)
                if adapter is not None:
                    pcbi_bridge_param_ids.update(id(p) for p in adapter.parameters())
            macro_gain = getattr(point_head, 'macro_gain', None)
            if isinstance(macro_gain, nn.Parameter):
                pcbi_bridge_param_ids.add(id(macro_gain))
        LOGGER.info(
            f'PCBI bridge-only warm-up: {pcbi_bridge_only_epochs} epoch(s), '
            f'{len(pcbi_bridge_param_ids)} trainable adapter tensor(s)')


    p2f_strict_isolation = (
        os.getenv('P2F_STRICT_ISOLATION', '0').strip().lower() in
        {'1', 'true', 'yes', 'on'})
    p2f_detect_head_only = (
        os.getenv('P2F_DETECT_HEAD_ONLY', '0').strip().lower() in
        {'1', 'true', 'yes', 'on'})
    p2f_train_stage = os.getenv('P2F_TRAIN_STAGE', '').strip().lower()
    if sum((p2f_strict_isolation, p2f_detect_head_only,
            bool(p2f_train_stage))) > 1:
        raise ValueError(
            'P2F_STRICT_ISOLATION, P2F_DETECT_HEAD_ONLY and '
            'P2F_TRAIN_STAGE are mutually exclusive')
    p2f_trainable_param_ids = set()
    if p2f_strict_isolation:
        p2f_model = de_parallel(model)
        if not bool(getattr(p2f_model, 'yaml', {}).get('p2f_peripheral', False)):
            raise ValueError('P2F_STRICT_ISOLATION requires a p2f_* model config')
        for layer_idx in range(24, 34):
            p2f_trainable_param_ids.update(
                id(p) for p in p2f_model.model[layer_idx].parameters())
        for module in p2f_model.modules():
            if module.__class__.__name__ in {
                    'PeripheralPointFourierFusion', 'FTInfer'}:
                p2f_trainable_param_ids.update(id(p) for p in module.parameters())
        LOGGER.info(
            'P2F strict isolation: backbone, Detect and PointResponse frozen; '
            f'{len(p2f_trainable_param_ids)} Fourier tensor(s) trainable')
    detect_head_param_ids = set()
    if p2f_detect_head_only:
        detect_model = de_parallel(model)
        for module in detect_model.modules():
            if module.__class__.__name__ == 'Detect':
                detect_head_param_ids.update(id(p) for p in module.parameters())
        if not detect_head_param_ids:
            raise ValueError('P2F_DETECT_HEAD_ONLY could not find a Detect module')
        LOGGER.info(
            'P2F Detect-head recovery: backbone, PointResponse and Fourier frozen; '
            f'{len(detect_head_param_ids)} Detect tensor(s) trainable')
    yolo_base_param_ids = set()
    yolo_base_active_layers = set()
    if p2f_train_stage == 'yolo_base':
        yolo_model = de_parallel(model)
        model_yaml = getattr(yolo_model, 'yaml', {})
        yolo_base_layers = model_yaml.get('yolo_base_layers', None)
        if yolo_base_layers is None and bool(model_yaml.get('p2f_peripheral', False)):


            yolo_base_layers = 24
        if yolo_base_layers is None:
            raise ValueError(
                'P2F_TRAIN_STAGE=yolo_base requires yolo_base_layers in the model config')
        yolo_base_layers = int(yolo_base_layers)
        if not 1 <= yolo_base_layers <= len(yolo_model.model):
            raise ValueError(
                f'Invalid yolo_base_layers={yolo_base_layers}; '
                f'model contains {len(yolo_model.model)} layers')


        yolo_base_active_layers.update(range(yolo_base_layers))
        for layer_idx in yolo_base_active_layers:
            yolo_base_param_ids.update(
                id(p) for p in yolo_model.model[layer_idx].parameters())
        detect_indices = []
        for layer_idx, module in enumerate(yolo_model.model):
            if module.__class__.__name__ == 'Detect':
                detect_indices.append(layer_idx)
                yolo_base_active_layers.add(layer_idx)
                yolo_base_param_ids.update(id(p) for p in module.parameters())
        if not detect_indices:
            raise ValueError('P2F yolo_base stage could not find a Detect module')
        LOGGER.info(
            'P2F stage A/yolo_base: train backbone + YOLO neck + Detect; '
            'freeze PointResponse/Fourier, '
            f'{len(yolo_base_param_ids)} tensor(s) trainable, '
            f'Detect layer(s)={detect_indices}')
    fourier_candidate_param_ids = set()
    fourier_candidate_active_layers = set()
    if p2f_train_stage == 'fourier_candidate_backbone':
        candidate_model = de_parallel(model)
        model_yaml = getattr(candidate_model, 'yaml', {})
        shared_span = model_yaml.get(
            'fourier_candidate_shared_layers', (5, 15))
        feature_span = model_yaml.get('fourier_feature_layers', (24, 34))
        if (not isinstance(shared_span, (tuple, list)) or
                len(shared_span) != 2):
            raise ValueError(
                'fourier_candidate_shared_layers must be '
                '[start, end_exclusive]')
        if (not isinstance(feature_span, (tuple, list)) or
                len(feature_span) != 2):
            raise ValueError(
                'fourier_feature_layers must be [start, end_exclusive]')
        shared_start, shared_end = map(int, shared_span)
        feature_start, feature_end = map(int, feature_span)
        if not 0 <= shared_start < shared_end <= len(candidate_model.model):
            raise ValueError(
                f'Invalid shared candidate span [{shared_start}, {shared_end}); '
                f'model contains {len(candidate_model.model)} layers')
        if not 0 <= feature_start < feature_end <= len(candidate_model.model):
            raise ValueError(
                f'Invalid Fourier feature span [{feature_start}, {feature_end}); '
                f'model contains {len(candidate_model.model)} layers')
        fourier_candidate_active_layers.update(
            range(shared_start, shared_end))
        fourier_candidate_active_layers.update(
            range(feature_start, feature_end))


        detach_boundaries = 0
        for layer_idx in range(feature_start, feature_end):
            module = candidate_model.model[layer_idx]
            if module.__class__.__name__ == 'DetachedConv':
                module.detach_input = False
                detach_boundaries += 1
            elif module.__class__.__name__ == 'ConcatDetachSkip':
                module.detach_skip = False
                detach_boundaries += 1
        for layer_idx in fourier_candidate_active_layers:
            fourier_candidate_param_ids.update(
                id(parameter)
                for parameter in candidate_model.model[layer_idx].parameters())
        proposal_indices = []
        for layer_idx, module in enumerate(candidate_model.model):
            if module.__class__.__name__ == 'FourierProposal':
                proposal_indices.append(layer_idx)
                fourier_candidate_active_layers.add(layer_idx)
                fourier_candidate_param_ids.update(
                    id(parameter) for parameter in module.parameters())
        if (len(proposal_indices) != 1 or
                not fourier_candidate_param_ids or
                detach_boundaries != 4):
            raise ValueError(
                'P2F fourier_candidate_backbone requires one FourierProposal '
                'and four detachable Fourier boundaries; found '
                f'proposal_layers={proposal_indices}, '
                f'boundaries={detach_boundaries}')
        LOGGER.info(
            'Fourier candidate pretrain: fixed shallow point-input layers and '
            f'Point/FTInfer; train shared layers [{shared_start}, {shared_end}), '
            f'Fourier layers [{feature_start}, {feature_end}) + '
            f'FourierProposal layer {proposal_indices[0]} '
            f'({len(fourier_candidate_param_ids)} parameter tensors)')
    point_only_param_ids = set()
    point_only_active_layers = set()
    detect_temporal_param_ids = set()
    detect_temporal_active_layers = set()
    if p2f_train_stage == 'detect_temporal':
        detect_temporal_model = de_parallel(model)
        detect_indices = []
        adapter_count = 0
        for layer_idx, module in enumerate(detect_temporal_model.model):
            if module.__class__.__name__ == 'Detect':
                detect_indices.append(layer_idx)
                detect_temporal_active_layers.add(layer_idx)
                detect_temporal_param_ids.update(
                    id(p) for p in module.parameters())
                adapter_count += len(getattr(module, 'temporal_adapters', ()))
        if not detect_temporal_param_ids or adapter_count != 3:
            raise ValueError(
                'P2F detect_temporal stage requires a Detect module with '
                f'exactly three temporal adapters; found {adapter_count}')
        LOGGER.info(
            'Detect-temporal stage: freeze backbone, PointResponse and Fourier; '
            'train Detect prediction convolutions + causal P3/P4/P5 adapters, '
            f'{len(detect_temporal_param_ids)} tensor(s) trainable, '
            f'Detect layer(s)={detect_indices}')
    if p2f_train_stage in {'point_only', 'point_denoise'}:
        point_model = de_parallel(model)
        for layer_idx, module in enumerate(point_model.model):
            if module.__class__.__name__ == 'BioCountInfer':
                point_only_active_layers.add(layer_idx)
                point_only_param_ids.update(id(p) for p in module.parameters())
        if not point_only_param_ids:
            raise ValueError(
                f'P2F {p2f_train_stage} stage could not find BioCountInfer')
        LOGGER.info(
            f'Point stage={p2f_train_stage}: train BioCountInfer only, '
            f'{len(point_only_param_ids)} tensor(s) trainable')
    point_motion_param_ids = set()
    if p2f_train_stage == 'point_motion_gate':
        motion_model = de_parallel(model)
        point_head = motion_model.get_module_byname('BioCountInfer')
        temporal_adapter = getattr(point_head, 'temporal_adapter', None)
        if (temporal_adapter is None or
                temporal_adapter.__class__.__name__ !=
                'ExplicitMotionPointTemporalAdapter'):
            raise ValueError(
                'P2F point_motion_gate requires BioCountInfer with '
                'ExplicitMotionPointTemporalAdapter')
        point_motion_param_ids.update(
            id(parameter) for parameter in temporal_adapter.parameters())
        if not point_motion_param_ids:
            raise ValueError(
                'P2F point_motion_gate found no temporal parameters')
        LOGGER.info(
            'Explicit point-motion stage: freeze spatial BioCountInfer and '
            f'all other branches; train {len(point_motion_param_ids)} '
            'motion-gate parameter tensors')
    point_motion_refine_param_ids = set()
    if p2f_train_stage == 'point_motion_refine':
        refine_model = de_parallel(model)
        point_head = refine_model.get_module_byname('BioCountInfer')
        temporal_adapter = getattr(point_head, 'temporal_adapter', None)
        cls_head = getattr(point_head, 'cls_head', None)
        if (temporal_adapter is None or
                temporal_adapter.__class__.__name__ !=
                'ExplicitMotionPointTemporalAdapter'):
            raise ValueError(
                'P2F point_motion_refine requires BioCountInfer with '
                'ExplicitMotionPointTemporalAdapter')
        if cls_head is None:
            raise ValueError(
                'P2F point_motion_refine requires BioCountInfer.cls_head')
        if not bool(getattr(point_head, 'detach_features', False)):
            raise ValueError(
                'P2F point_motion_refine requires detached point features '
                'to protect the backbone/neck')
        point_motion_refine_param_ids.update(
            id(parameter) for parameter in temporal_adapter.parameters())
        point_motion_refine_param_ids.update(
            id(parameter) for parameter in cls_head.parameters())
        if not point_motion_refine_param_ids:
            raise ValueError(
                'P2F point_motion_refine found no trainable parameters')
        LOGGER.info(
            'Point motion-classification refinement: backbone, Fourier, '
            'decoder and regression heads frozen; train '
            f'{len(point_motion_refine_param_ids)} temporal/cls tensor(s)')
    dual_temporal_param_ids = set()
    if p2f_train_stage == 'dual_temporal_only':
        temporal_model = de_parallel(model)
        point_adapters = 0
        fourier_adapters = 0
        point_head = temporal_model.get_module_byname('BioCountInfer')
        point_adapter = getattr(point_head, 'temporal_adapter', None)
        if point_adapter is not None:
            point_adapters = 1
            dual_temporal_param_ids.update(
                id(parameter) for parameter in point_adapter.parameters())
        fourier_head = temporal_model.get_module_byname('FTInfer')
        for adapter in getattr(fourier_head, 'temporal_adapters', ()):
            fourier_adapters += 1
            dual_temporal_param_ids.update(
                id(parameter) for parameter in adapter.parameters())
        if not dual_temporal_param_ids or point_adapters != 1 or fourier_adapters != 3:
            raise ValueError(
                'P2F dual_temporal_only requires one Point and exactly three '
                f'Fourier temporal adapters; found {point_adapters} and '
                f'{fourier_adapters}')
        LOGGER.info(
            'Dual-temporal warm-up: freeze every spatial parameter; train '
            f'{len(dual_temporal_param_ids)} temporal tensor(s) from one Point '
            'adapter and three Fourier scale adapters')
    outward_p2f_param_ids = set()
    if p2f_train_stage in {
            'point_to_fourier_outward',
             'point_to_fourier_shape_refine',
             'point_to_fourier_stable',
             'point_to_fourier_peripheral',
             'fourier_contour_expand',
             'fourier_directional_geometry',
             'fourier_directional_oracle',
             'fourier_point_rescue',
             'fourier_hybrid_match',
             'point_to_fourier_joint'}:
        outward_model = de_parallel(model)
        bridge_count = 0
        for module in outward_model.modules():
            if module.__class__.__name__ in {
                    'OutwardPointFourierFusion',
                    'CandidateDirectionalFourierBridge'}:
                bridge_count += 1
                outward_p2f_param_ids.update(
                    id(parameter) for parameter in module.parameters())
            elif module.__class__.__name__ == 'FTInfer':
                outward_p2f_param_ids.update(
                    id(parameter) for parameter in module.parameters())
            elif (p2f_train_stage in {
                    'point_to_fourier_shape_refine',
                    'point_to_fourier_joint'} and
                  module.__class__.__name__ == 'FourierProposal'):
                outward_p2f_param_ids.update(
                    id(parameter) for parameter in module.parameters())
        if bridge_count != 3 or not outward_p2f_param_ids:
            raise ValueError(
                f'P2F {p2f_train_stage} requires exactly three '
                f'OutwardPointFourierFusion modules; found {bridge_count}')
        LOGGER.info(
            f'Outward Point->Fourier stage={p2f_train_stage}: frozen '
            'Point/backbone; train three bridges + FTInfer' +
            (' + FourierProposal' if p2f_train_stage in {
                'point_to_fourier_shape_refine',
                'point_to_fourier_joint'} else '') +
            f' ({len(outward_p2f_param_ids)} '
            'parameter tensors)')
    feature_bridge_param_ids = set()
    feature_bridge_active_layers = set()
    if p2f_train_stage == 'feature_bridge':
        feature_model = de_parallel(model)
        if not bool(getattr(feature_model, 'yaml', {}).get(
                'point_fourier_feature_bridge', False)):
            raise ValueError(
                'P2F_TRAIN_STAGE=feature_bridge requires a '
                'point_fourier_feature_bridge model config')


        feature_bridge_active_layers.update(range(24, 34))
        feature_bridge_active_layers.update({36, 37, 38, 39})
        for layer_idx in feature_bridge_active_layers:
            if layer_idx < len(feature_model.model):
                feature_bridge_param_ids.update(
                    id(p) for p in feature_model.model[layer_idx].parameters())
        LOGGER.info(
            'Feature-bridge stage: freeze backbone/YOLO/PointResponse; '
            'train Fourier neck + YOLO-Point-Fourier gates + FTInfer, '
            f'{len(feature_bridge_param_ids)} tensor(s) trainable')
    elif p2f_train_stage == 'feature_joint':
        joint_model = de_parallel(model)
        if not bool(getattr(joint_model, 'yaml', {}).get(
                'point_fourier_feature_bridge', False)):
            raise ValueError(
                'P2F_TRAIN_STAGE=feature_joint requires a '
                'point_fourier_feature_bridge model config')
        for module in joint_model.modules():
            if module.__class__.__name__ == 'YoloPointFourierFeatureBridge':
                module.detach_point = False
                module.detach_yolo = False
        LOGGER.info(
            'Feature-joint stage: all three heads and feature bridges trainable; '
            'Fourier cross-gradients scaled to Point=0.10 and YOLO=0.05')

    for epoch in range(start_epoch, epochs):
        bridge_only = (
            pcbi_bridge_only_epochs > 0 and
            (epoch - start_epoch) < pcbi_bridge_only_epochs)
        for parameter in model.parameters():
            if p2f_strict_isolation:
                parameter.requires_grad_(id(parameter) in p2f_trainable_param_ids)
            elif p2f_detect_head_only:
                parameter.requires_grad_(id(parameter) in detect_head_param_ids)
            elif p2f_train_stage == 'yolo_base':
                parameter.requires_grad_(id(parameter) in yolo_base_param_ids)
            elif p2f_train_stage == 'fourier_candidate_backbone':
                parameter.requires_grad_(
                    id(parameter) in fourier_candidate_param_ids)
            elif p2f_train_stage == 'detect_temporal':
                parameter.requires_grad_(
                    id(parameter) in detect_temporal_param_ids)
            elif p2f_train_stage in {'point_only', 'point_denoise'}:
                parameter.requires_grad_(id(parameter) in point_only_param_ids)
            elif p2f_train_stage == 'point_motion_gate':
                parameter.requires_grad_(
                    id(parameter) in point_motion_param_ids)
            elif p2f_train_stage == 'point_motion_refine':
                parameter.requires_grad_(
                    id(parameter) in point_motion_refine_param_ids)
            elif p2f_train_stage == 'dual_temporal_only':
                parameter.requires_grad_(
                    id(parameter) in dual_temporal_param_ids)
            elif p2f_train_stage in {
                    'point_to_fourier_outward',
                     'point_to_fourier_shape_refine',
                     'point_to_fourier_stable',
                     'point_to_fourier_peripheral',
                     'fourier_contour_expand',
                     'fourier_directional_geometry',
                     'fourier_directional_oracle',
                     'fourier_point_rescue',
                     'fourier_hybrid_match',
                     'point_to_fourier_joint'}:
                parameter.requires_grad_(
                    id(parameter) in outward_p2f_param_ids)
            elif p2f_train_stage == 'feature_bridge':
                parameter.requires_grad_(id(parameter) in feature_bridge_param_ids)
            else:
                parameter.requires_grad_(
                    (not bridge_only) or id(parameter) in pcbi_bridge_param_ids)
        if pcbi_bridge_only_epochs > 0 and epoch == start_epoch + pcbi_bridge_only_epochs:
            LOGGER.info('PCBI bridge-only warm-up finished; all task parameters unfrozen')
        model.train()
        if p2f_strict_isolation:


            active_p2f_layers = set(range(24, 34)) | {35, 36, 37, 39}
            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in active_p2f_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_detect_head_only:


            for layer in de_parallel(model).model:
                if layer.__class__.__name__ != 'Detect':
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_train_stage == 'yolo_base':


            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in yolo_base_active_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_train_stage == 'fourier_candidate_backbone':


            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in fourier_candidate_active_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_train_stage == 'detect_temporal':


            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in detect_temporal_active_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_train_stage in {'point_only', 'point_denoise'}:


            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in point_only_active_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()
        elif p2f_train_stage == 'point_motion_gate':


            for child in de_parallel(model).modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()
        elif p2f_train_stage == 'point_motion_refine':


            for child in de_parallel(model).modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()
        elif p2f_train_stage == 'dual_temporal_only':


            for child in de_parallel(model).modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()
        elif p2f_train_stage in {
                'point_to_fourier_outward',
                 'point_to_fourier_shape_refine',
                 'point_to_fourier_stable',
                 'point_to_fourier_peripheral',
                 'fourier_contour_expand',
                 'fourier_directional_geometry',
                 'fourier_directional_oracle',
                 'fourier_point_rescue',
                 'fourier_hybrid_match',
                 'point_to_fourier_joint'}:


            for child in de_parallel(model).modules():
                if isinstance(child, nn.BatchNorm2d):
                    child.eval()
        elif p2f_train_stage == 'feature_bridge':

            for layer_idx, layer in enumerate(de_parallel(model).model):
                if layer_idx not in feature_bridge_active_layers:
                    for child in layer.modules():
                        if isinstance(child, nn.BatchNorm2d):
                            child.eval()


        if opt.image_weights:

            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc


            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)


            neg_set = iw == 0
            neg_count = iw[neg_set].shape[0]
            neg_alpha = hyp.get('neg_alpha', 0.03)


            if neg_count > 0:
                nw = neg_alpha * iw.sum(0) / ((1 - neg_alpha)*neg_count)
                iw[neg_set] += nw
                neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
                if start_epoch==epoch:
                    print("neg_alpha_val=",neg_alpha_val)

            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)


        mloss = torch.zeros(5, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('%10s' * 9) % (
            'Epoch', 'gpu_mem', 'box', 'obj', 'ft_xy', 'ft_coef',
            'point', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, ncols=max(shutil.get_terminal_size().columns - 10, 10), dynamic_ncols=False)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:


            if imgs.ndimension() == 5:

                b, t, c, h, img_w = imgs.shape
                imgs = imgs.view(b * t, c, h, img_w)
            else:

                bt, c, h, img_w = imgs.shape
                t = 5
                b = bt // t


            if epoch == start_epoch and i < 3:
                debug_dir = save_dir / 'vis_loader_check_raw'
                if not os.path.exists(debug_dir): os.makedirs(debug_dir)

                print(f"\n?[PROBE] Visualizing RAW Batch {i} to {debug_dir}...")


                vis_imgs = imgs.clone().detach().cpu().numpy()


                vis_imgs = vis_imgs.astype(np.uint8).transpose(0, 2, 3, 1)

                vis_targets = targets.clone().detach().cpu().numpy()


                def reconstruct_polygon(a0, c0, ft_coefs, num_points=50):
                    t_val = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                    x = np.full_like(t_val, a0)
                    y = np.full_like(t_val, c0)
                    terms = len(ft_coefs) // 4
                    for k in range(1, terms + 1):
                        idx = (k - 1) * 4
                        ak, bk, ck, dk = ft_coefs[idx:idx+4]
                        x += ak * np.cos(k * t_val) + bk * np.sin(k * t_val)
                        y += ck * np.cos(k * t_val) + dk * np.sin(k * t_val)
                    pts = np.stack((x, y), axis=1).astype(np.int32)
                    return pts.reshape((-1, 1, 2))

                for clip_idx in range(b):
                    for frame_offset in range(t):
                        global_frame_idx = clip_idx * t + frame_offset


                        curr_boxes = vis_targets[vis_targets[:, 0] == global_frame_idx]
                        img_draw = vis_imgs[global_frame_idx].copy()
                        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                        h_img, w_img = img_draw.shape[:2]

                        for box in curr_boxes:

                            cx, cy, w_box, h_box = box[2], box[3], box[4], box[5]
                            x1 = int((cx - w_box/2) * w_img)
                            y1 = int((cy - h_box/2) * h_img)
                            x2 = int((cx + w_box/2) * w_img)
                            y2 = int((cy + h_box/2) * h_img)

                            color = (0, 255, 0) if frame_offset == t - 1 else (0, 255, 255)
                            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(img_draw, (int(cx * w_img), int(cy * h_img)), 3, (0, 0, 255), -1)


                            if len(box) > 8:
                                a0 = box[6] * w_img
                                c0 = box[7] * h_img
                                raw_coefs = box[8:]
                                ft_coefs = np.zeros_like(raw_coefs)
                                ft_coefs[0::4] = raw_coefs[0::4] * w_img
                                ft_coefs[1::4] = raw_coefs[1::4] * w_img
                                ft_coefs[2::4] = raw_coefs[2::4] * h_img
                                ft_coefs[3::4] = raw_coefs[3::4] * h_img

                                poly_pts = reconstruct_polygon(a0, c0, ft_coefs)
                                poly_color = (255, 0, 255)
                                cv2.polylines(img_draw, [poly_pts], isClosed=True, color=poly_color, thickness=1)

                        cv2.imwrite(str(debug_dir / f"batch_{i}_clip_{clip_idx}_frame_{frame_offset}.jpg"), img_draw)


            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255


            if ni <= nw:
                xi = [0, nw]

                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):

                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])


            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)


            with torch.amp.autocast('cuda', enabled=cuda):


                if oracle_peripheral_gt:
                    GLOBAL_ORACLE_PERIPHERAL_DICT[imgs.device] = {
                        'targets': targets.to(imgs.device, non_blocking=True).detach()
                    }
                else:
                    GLOBAL_ORACLE_PERIPHERAL_DICT.pop(imgs.device, None)
                pred = model(imgs)


                if len(imgs) != len(paths):
                    seq_len = len(imgs) // len(paths)

                    last_indices = torch.arange(seq_len - 1, len(imgs), seq_len, device=device)


                    def filter_last(p):
                        if isinstance(p, dict):
                            return {k: filter_last(v) for k, v in p.items()}
                        if isinstance(p, (list, tuple)):
                            return type(p)(filter_last(x) for x in p)
                        elif isinstance(p, torch.Tensor):
                            return p[last_indices]
                        return p


                    if point_temporal_many_to_many:


                        if not (isinstance(pred, list) and pred and
                                isinstance(pred[-1], dict)):
                            raise RuntimeError(
                                'Temporal point training expects the final model '
                                'output to be the BioCountInfer dictionary')
                        pred = [filter_last(item) for item in pred[:-1]] + [pred[-1]]
                    else:
                        pred = filter_last(pred)


                g_map = model.module.guidance_map if hasattr(model, 'module') else getattr(model, 'guidance_map', None)

                compute_loss.epoch = epoch
                loss, loss_items = compute_loss(pred, targets.to(device), fold=fold, mask_dir=model.mask_dir if not hasattr(model, 'module') else model.module.mask_dir, paths=paths)

                log_point_diagnostics = (
                    p2f_train_stage in {
                        'point_motion_gate', 'point_motion_refine'} or
                    float(hyp.get(
                        'biopoint_candidate_hard_positive', 0.0)) > 0.0)
                if i == 0 and log_point_diagnostics:
                    motion_diag = getattr(
                        compute_loss, 'last_point_motion', None)
                    if isinstance(motion_diag, dict):
                        LOGGER.info(
                            'Point motion gate: '
                            f"loss={float(motion_diag['loss']):.6g}, "
                            f"mean={float(motion_diag['gate_mean']):.6g}, "
                            f"max={float(motion_diag['gate_max']):.6g}, "
                            f"hard_peak_loss={float(motion_diag.get('hard_peak_loss', 0.0)):.6g}, "
                            f"hard_peaks/frame={float(motion_diag.get('hard_peak_count', 0.0)):.3f}, "
                            f"candidate_pos_loss={float(motion_diag.get('candidate_hard_positive_loss', 0.0)):.6g}, "
                            f"candidate_pos/frame={float(motion_diag.get('candidate_hard_positive_count', 0.0)):.3f}, "
                            f"candidate_prior_mean={float(motion_diag.get('candidate_prior_mean', 0.0)):.6g}, "
                            f"candidate_prior_max={float(motion_diag.get('candidate_prior_max', 0.0)):.6g}, "
                            f"candidate_gain={float(motion_diag.get('candidate_recall_gain', 0.0)):.6g}")
                        if p2f_train_stage == 'point_motion_refine':
                            hard_peak_file = (
                                save_dir / 'point_hard_peak_history.csv')
                            if not hard_peak_file.exists():
                                hard_peak_file.write_text(
                                    'epoch,hard_peak_loss,hard_peaks_per_frame\n',
                                    encoding='utf-8')
                            with hard_peak_file.open(
                                    'a', encoding='utf-8') as stream:
                                stream.write(
                                    f"{epoch},"
                                    f"{float(motion_diag.get('hard_peak_loss', 0.0)):.8f},"
                                    f"{float(motion_diag.get('hard_peak_count', 0.0)):.8f}\n")

                log_fourier_geometry = any(float(hyp.get(key, 0.0)) > 0.0 for key in (
                    'ft_cover_gt', 'ft_cover_pred', 'ft_curve', 'ft_area',
                    'ft_shape', 'ft_harmonic_shape', 'ft_directional'))
                if i == 0 and log_fourier_geometry:
                    alignment = getattr(
                        compute_loss, 'last_fourier_alignment', None)
                    if isinstance(alignment, dict):
                        LOGGER.info(
                            'Fourier coverage alignment: '
                            f"seq_len={alignment['seq_len']}, "
                            f"point_micro(all frames)="
                            f"{alignment['point_micro_count']}, "
                            f"fourier_micro(last frame)="
                            f"{alignment['fourier_micro_count']}, "
                            f"gt_cover={float(alignment['cover_loss']):.6g}, "
                            f"pred_cover="
                            f"{float(alignment['pred_cover_loss']):.6g}, "
                            f"shape_directional="
                            f"{float(alignment['directional_loss']):.6g}")

                if hasattr(model, 'module'):
                    model.module.guidance_map = None
                else:
                    model.guidance_map = None
                if RANK != -1:
                    loss *= WORLD_SIZE
                if opt.quad:
                    loss *= 4.


            loss_is_safe = bool(torch.isfinite(loss.detach()).all())
            if loss_is_safe:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

            grad_is_safe = loss_is_safe
            bad_grad_name = None if loss_is_safe else 'total_loss'
            if loss_is_safe:
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        grad_is_safe = False
                        bad_grad_name = name
                        break


            if (grad_is_safe and point_backbone_grad and
                    epoch == start_epoch and i == 0):
                gradient_model = de_parallel(model)

                def module_gradient_norm(modules):
                    squared_norm = 0.0
                    tensor_count = 0
                    for gradient_module in modules:
                        for parameter in gradient_module.parameters():
                            if parameter.grad is not None:
                                squared_norm += float(
                                    parameter.grad.detach().float().square().sum())
                                tensor_count += 1
                    return squared_norm ** 0.5, tensor_count

                source_layers = list(gradient_model.model[:5])
                source_norm, source_tensors = module_gradient_norm(
                    source_layers)
                point_norm, point_tensors = module_gradient_norm(
                    [point_gradient_head])
                LOGGER.warning(
                    'Point-backbone gradient check (first batch): '
                    f'shared_layers_0_4={source_norm:.6g} '
                    f'({source_tensors} tensors), '
                    f'BioCountInfer={point_norm:.6g} '
                    f'({point_tensors} tensors)')
                gradient_check_file = (
                    save_dir / 'point_backbone_gradient_check.txt')
                gradient_check_file.write_text(
                    'detach_features=false\n'
                    f'shared_layers_0_4_grad_norm={source_norm:.12g}\n'
                    f'shared_layers_0_4_grad_tensors={source_tensors}\n'
                    f'point_head_grad_norm={point_norm:.12g}\n'
                    f'point_head_grad_tensors={point_tensors}\n',
                    encoding='utf-8')

            if grad_is_safe:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                print(f"[SKIP BATCH] NaN/Inf gradient detected at {bad_grad_name}; "
                      f"optimizer step skipped (AMP scale={scaler.get_scale():.1f}).")
                if loss_is_safe:


                    scaler.update()
                else:


                    scaler.update(new_scale=max(float(scaler.get_scale()) * 0.5, 1.0))

            optimizer.zero_grad(set_to_none=True)


            if ema and grad_is_safe:
                ema.update(model)


                last_opt_step = ni


            if RANK in [-1, 0]:


                for j, loss in enumerate(loss_items):
                    if not torch.isnan(loss):
                        mloss[j] = (mloss[j] * i + loss) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'


                pbar.set_description(('%10s' * 2 + '%10.4g' * 6 + '%10s') % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], f'{imgs.shape[-1]}x{imgs.shape[-2]}'))


                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, opt.sync_bn)


        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()
        if RANK in [-1, 0]:

            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            active_model = model.module if hasattr(model, 'module') else model
            formation_macro_detect = bool(getattr(active_model, 'yaml', {}).get('formation_macro_detect', False))
            fourier_proposal_mode = bool(
                getattr(active_model, 'yaml', {}).get('fourier_proposal', False))
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            fi = np.zeros((1),dtype=np.float32)
            if epoch >= data_dict.get('val_epoch', 0) and (not noval or final_epoch):
                T = data_dict.get('val_epoch_T', 1)
                if (epoch+1) % T == 0:
                    results, maps, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            iou_thres=0.2 if fourier_proposal_mode else (0.45 if formation_macro_detect else (0.2 if ft_coef > 0 else 0.45)),
                                            ab_thres=3.0,
                                            imgsz=imgsz,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            fold=fold,
                                            pts=pts,
                                            nms_polygon=1 if fourier_proposal_mode else (0 if formation_macro_detect else 1),


                                            polygon=not historical_t2_validation,
                                            legacy_t2_metrics=historical_t2_validation
                                            )


                    fi = fitness(np.array(results).reshape(1, -1))
                    if fi > best_fitness:
                        best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)


            if bool(getattr(active_model, 'yaml', {}).get('pcbi_enabled', False)):
                gate_values = []
                for module in active_model.modules():
                    if module.__class__.__name__ == 'PointFeatureGate':
                        gate_values.append(float(
                            module.max_gain * torch.sigmoid(module.gain.detach())))
                point_head = active_model.get_module_byname('BioCountInfer')
                macro_gain = getattr(point_head, 'macro_gain', None)
                contour_gain = (
                    float(torch.sigmoid(macro_gain.detach()))
                    if (isinstance(macro_gain, nn.Parameter) and
                        bool(getattr(active_model, 'yaml', {}).get(
                            'point_contour_prior', True))) else 0.0)
                gate_file = save_dir / 'pcbi_gate_history.csv'
                if not gate_file.exists():
                    gate_file.write_text(
                        'epoch,contour_to_point,detect_p3,detect_p4,detect_p5,'
                        'fourier_p3,fourier_p4,fourier_p5\n', encoding='utf-8')
                padded_gates = (gate_values + [0.0] * 6)[:6]
                with gate_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},{contour_gain:.8f},' +
                        ','.join(f'{value:.8f}' for value in padded_gates) + '\n')
                LOGGER.info(
                    f'PCBI gains: C2P={contour_gain:.4f}, '
                    f'P2C={[round(value, 4) for value in padded_gates]}')

            if bool(getattr(active_model, 'yaml', {}).get('p2f_peripheral', False)):
                p2f_gains = []
                for module in active_model.modules():
                    if module.__class__.__name__ == 'PeripheralPointFourierFusion':
                        p2f_gains.append(float(
                            module.max_gain * torch.sigmoid(module.gain.detach())))
                gain_file = save_dir / 'p2f_gain_history.csv'
                if not gain_file.exists():
                    gain_file.write_text(
                        'epoch,fourier_p3,fourier_p4,fourier_p5\n', encoding='utf-8')
                padded_p2f = (p2f_gains + [0.0] * 3)[:3]
                with gain_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},' + ','.join(
                            f'{value:.8f}' for value in padded_p2f) + '\n')
                LOGGER.info(
                    f'P2F peripheral gains: '
                    f'{[round(value, 4) for value in padded_p2f]}')

            if bool(getattr(active_model, 'yaml', {}).get(
                    'point_fourier_feature_bridge', False)):
                feature_rows = []
                for module in active_model.modules():
                    if module.__class__.__name__ == 'YoloPointFourierFeatureBridge':
                        feature_rows.append((
                            float(module.max_point_gain * torch.sigmoid(
                                module.point_gain.detach())),
                            float(module.max_yolo_gain * torch.sigmoid(
                                module.yolo_gain.detach())),
                            float(module.last_stats.get(
                                'gate_mean', torch.tensor(float('nan'))))))
                feature_rows = (feature_rows + [(0.0, 0.0, 0.0)] * 3)[:3]
                feature_file = save_dir / 'feature_bridge_history.csv'
                if not feature_file.exists():
                    feature_file.write_text(
                        'epoch,p3_point,p3_yolo,p3_gate,p4_point,p4_yolo,'
                        'p4_gate,p5_point,p5_yolo,p5_gate\n', encoding='utf-8')
                with feature_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},' + ','.join(
                            f'{value:.8f}' for row in feature_rows for value in row) + '\n')
                LOGGER.info(
                    'Feature bridge [point,yolo,gate] P3-P5: ' +
                    str([[round(value, 4) for value in row]
                         for row in feature_rows]))

            if bool(getattr(active_model, 'yaml', {}).get(
                    'outward_point_fourier_bridge', False)):
                outward_rows = []
                for module in active_model.modules():
                    if module.__class__.__name__ in {
                            'OutwardPointFourierFusion',
                            'CandidateDirectionalFourierBridge'}:
                        gain = float(
                            module.max_gain * torch.sigmoid(
                                module.gain_logit.detach()))
                        outward_rows.append((
                            gain,
                            float(module.last_stats.get(
                                'external_mean', torch.tensor(float('nan')))),
                            float(module.last_stats.get(
                                'external_max', torch.tensor(float('nan')))),
                            float(module.last_stats.get(
                                'external_active',
                                torch.tensor(float('nan'))))))
                outward_rows = (
                    outward_rows + [(0.0, 0.0, 0.0, 0.0)] * 3)[:3]
                outward_file = save_dir / 'outward_p2f_history.csv'
                if not outward_file.exists():
                    outward_file.write_text(
                        'epoch,p3_gain,p3_external_mean,p3_external_max,'
                        'p3_external_active,p4_gain,p4_external_mean,'
                        'p4_external_max,p4_external_active,p5_gain,'
                        'p5_external_mean,p5_external_max,'
                        'p5_external_active\n',
                        encoding='utf-8')
                with outward_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},' + ','.join(
                            f'{value:.8f}' for row in outward_rows
                            for value in row) + '\n')
                LOGGER.info(
                    'Outward P2F [gain,external_mean,external_max,'
                    'external_active] P3-P5: ' +
                    str([[round(value, 4) for value in row]
                         for row in outward_rows]))

            if bool(getattr(active_model, 'yaml', {}).get(
                    'oracle_peripheral_gt', False)):
                oracle_rows = []
                for module in active_model.modules():
                    if module.__class__.__name__ == 'OraclePeripheralFourierBridge':
                        oracle_rows.append((
                            float(module.max_gain * torch.sigmoid(
                                module.gain_logit.detach())),
                            float(module.last_stats.get(
                                'gate_mean', torch.tensor(float('nan')))),
                            float(module.last_stats.get(
                                'peripheral_max', torch.tensor(float('nan')))),
                            float(module.last_stats.get(
                                'peripheral_mean', torch.tensor(float('nan'))))))
                oracle_rows = (
                    oracle_rows + [(0.0, 0.0, 0.0, 0.0)] * 3)[:3]
                oracle_file = save_dir / 'oracle_bridge_history.csv'
                if not oracle_file.exists():
                    oracle_file.write_text(
                        'epoch,p3_gain,p3_gate,p3_peripheral_max,p3_peripheral_mean,'
                        'p4_gain,p4_gate,p4_peripheral_max,p4_peripheral_mean,'
                        'p5_gain,p5_gate,p5_peripheral_max,p5_peripheral_mean\n',
                        encoding='utf-8')
                with oracle_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},' + ','.join(
                            f'{value:.8f}' for row in oracle_rows for value in row) +
                        '\n')
                LOGGER.info(
                    'Oracle bridge [gain,gate,peripheral_max,peripheral_mean] '
                    'P3-P5: ' + str([
                        [round(value, 4) for value in row]
                        for row in oracle_rows]))

            if bool(getattr(active_model, 'yaml', {}).get(
                    'detect_temporal_enabled', False)):
                detect_temporal_gains = []
                for module in active_model.modules():
                    if module.__class__.__name__ == 'Detect':
                        for adapter in getattr(
                                module, 'temporal_adapters', ()):
                            detect_temporal_gains.append(float(
                                adapter.max_gain * torch.sigmoid(
                                    adapter.gain_logit.detach())))
                detect_temporal_gains = (
                    detect_temporal_gains + [0.0] * 3)[:3]
                temporal_file = save_dir / 'detect_temporal_history.csv'
                if not temporal_file.exists():
                    temporal_file.write_text(
                        'epoch,p3_gain,p4_gain,p5_gain\n', encoding='utf-8')
                with temporal_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},' + ','.join(
                            f'{value:.8f}'
                            for value in detect_temporal_gains) + '\n')
                LOGGER.info(
                    'Detect temporal gains P3-P5: ' +
                    str([round(value, 4)
                         for value in detect_temporal_gains]))

            if (bool(getattr(active_model, 'yaml', {}).get(
                    'point_temporal_enabled', False)) or
                    bool(getattr(active_model, 'yaml', {}).get(
                        'ft_temporal_enabled', False))):
                point_gain = 0.0
                fourier_gains = []
                point_head = active_model.get_module_byname('BioCountInfer')
                point_adapter = getattr(point_head, 'temporal_adapter', None)
                if point_adapter is not None:
                    if (hasattr(point_adapter, 'max_gain') and
                            hasattr(point_adapter, 'gain_logit')):
                        point_gain = float(
                            point_adapter.max_gain * torch.sigmoid(
                                point_adapter.gain_logit.detach()))
                    else:
                        point_gain = float(
                            point_adapter.last_stats.get(
                                'gate_mean', torch.tensor(float('nan'))))
                        motion_file = (
                            save_dir / 'point_motion_gate_history.csv')
                        if not motion_file.exists():
                            motion_file.write_text(
                                'epoch,gate_mean,gate_max,background_keep,'
                                'output_gate_mean,output_gate_min,'
                                'output_gate_max\n',
                                encoding='utf-8')
                        motion_row = (
                            float(point_adapter.last_stats.get(
                                'gate_mean', torch.tensor(float('nan')))),
                            float(point_adapter.last_stats.get(
                                'gate_max', torch.tensor(float('nan')))),
                            float(point_adapter.last_stats.get(
                                'background_keep',
                                torch.tensor(float('nan')))),
                            float(point_adapter.last_stats.get(
                                'output_gate_mean',
                                torch.tensor(float('nan')))),
                            float(point_adapter.last_stats.get(
                                'output_gate_min',
                                torch.tensor(float('nan')))),
                            float(point_adapter.last_stats.get(
                                'output_gate_max',
                                torch.tensor(float('nan')))))
                        with motion_file.open(
                                'a', encoding='utf-8') as stream:
                            stream.write(
                                f'{epoch},' +
                                ','.join(
                                    f'{value:.8f}'
                                    for value in motion_row) + '\n')
                        LOGGER.info(
                            'Explicit point motion '
                            '[gate_mean,gate_max,background_keep,'
                            'output_mean,output_min,output_max]: ' +
                            str([round(value, 4)
                                 for value in motion_row]))
                fourier_head = active_model.get_module_byname('FTInfer')
                for adapter in getattr(
                        fourier_head, 'temporal_adapters', ()):
                    fourier_gains.append(float(
                        adapter.max_gain * torch.sigmoid(
                            adapter.gain_logit.detach())))
                fourier_gains = (fourier_gains + [0.0] * 3)[:3]
                history_file = save_dir / 'point_fourier_temporal_history.csv'
                if not history_file.exists():
                    history_file.write_text(
                        'epoch,point_gain,ft_p3_gain,ft_p4_gain,ft_p5_gain\n',
                        encoding='utf-8')
                with history_file.open('a', encoding='utf-8') as stream:
                    stream.write(
                        f'{epoch},{point_gain:.8f},' +
                        ','.join(f'{value:.8f}' for value in fourier_gains) +
                        '\n')
                LOGGER.info(
                    'Point/Fourier temporal gains: '
                    f'Point={point_gain:.4f}, '
                    f'FT={[round(value, 4) for value in fourier_gains]}')


            if (not nosave) or (final_epoch and not evolve):
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'date': datetime.now().isoformat()}


                if time.time() - last_save_time > 300 or final_epoch:
                    torch.save(ckpt, last)
                    last_save_time = time.time()

                if fi > 0:
                    if best_fitness == fi:
                        torch.save(ckpt, best)


                        src,dst = str(save_dir / 'threshs.npy'),str(w / 'threshs.npy')

                        shutil.copy(src, dst)

                    if best_ap50 < results[2]:
                        best_ap50 = results[2]
                        torch.save(ckpt, best_ap)


                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

                del ckpt


            if RANK == -1 and stopper(epoch=epoch, fitness=fi):
                break


    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in best, best_ap:
            if f.exists():
                strip_optimizer(f)
                if f is best_ap:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(f, device).half(),
                                            conf_thres=0.001,
                                            iou_thres=0.2 if fourier_proposal_mode else (0.45 if formation_macro_detect else (0.2 if ft_coef > 0 else 0.45)),
                                            ab_thres=3.0,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            verbose=True,
                                            plots=True,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss,
                                            fold=fold,

                                            pts=pts,
                                            nms_polygon = 1 if fourier_proposal_mode else (0 if formation_macro_detect else 1),
                                            polygon=True
                                            )
                    if is_coco:
                        if mloss!=None:
                            callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        LOGGER.info(f"Results saved to \033[32m{colorstr('bold', save_dir)}\033[0m")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/infrared_e2.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/repvit_m0_9_t2.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'hyps/hyp.repvit_m0_9_t2.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument(
        '--imgsz', '--img', '--img-size', type=int, nargs='+',
        default=[640, 640], help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--cache_mosaic', type=int, default=512, help='--cache_mosaic')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')


    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')

    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=300, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--fold', type=int, default=1, help='fold angle')
    parser.add_argument('--augment', default=1, action='store_true', help='augment')
    parser.add_argument('--hm', action='store_true', default=False, help='ss')


    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
    parser.add_argument('--hull', type=int, default=1, help='hull')

    parser.add_argument('--plots', type=int, default=0, help='plot_labels')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()


    cli_tokens = sys.argv[1:]
    explicit_cli_dests = set()
    for action in parser._actions:
        if any(
                token == option or token.startswith(option + '=')
                for token in cli_tokens for option in action.option_strings):
            explicit_cli_dests.add(action.dest)
    explicit_cli_values = {
        dest: deepcopy(getattr(opt, dest))
        for dest in explicit_cli_dests if hasattr(opt, dest)
    }
    opt.fold = 2
    opt.data = 'data/infrared_e2.yaml'
    opt.cfg = 'models/repvit_m0_9_t2.yaml'
    opt.weights = ''
    opt.hyp = 'hyps/hyp.repvit_m0_9_t2.yaml'
    opt.imgsz = [640, 640]
    opt.epochs = int(os.getenv('EXPERIMENT_EPOCHS', '50'))
    opt.experiment_seed = int(os.getenv('EXPERIMENT_SEED', '0'))
    opt.cache_mosaic = 0
    opt.batch_size = 2
    opt.noautoanchor = True
    opt.save_period = 10
    opt.plots = 0
    opt.workers = 4
    opt.project = str(ROOT / 'runs/train')
    opt.name = 'repvit_m0_9_t2'
    for dest, value in explicit_cli_values.items():
        setattr(opt, dest, value)
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    return opt


def main(opt, callbacks=Callbacks()):

    if RANK in [-1, 0]:
        print_args(FILE.stem, opt)


    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        if not os.path.exists(ckpt):
            print(f'\033[91m{ckpt} not exists.\033[0m')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt_resume = argparse.Namespace(**yaml.safe_load(f))
            opt_resume.epochs = opt.epochs
            opt = opt_resume

        if(os.path.basename(ckpt)=='last.pt'):
            opt.weights = ckpt
        opt.resume = True if os.path.exists(opt.weights) else False
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        if os.path.exists(opt.data) and os.path.exists(opt.cfg):
            opt.data, opt.cfg, opt.hyp, opt.weights, opt.project =\
                check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            if opt.evolve:
                opt.project = str(ROOT / 'runs/evolve')
                opt.exist_ok, opt.resume = opt.resume, False
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))


        else:
            print(f'Dataset or model config does not exist: {opt.data}, {opt.cfg}')


    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
        if WORLD_SIZE > 1 and RANK == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()


    else:

        meta = {'lr0': (1, 1e-5, 1e-1),
                'lrf': (1, 0.01, 1.0),
                'momentum': (0.3, 0.6, 0.98),
                'weight_decay': (1, 0.0, 0.001),
                'warmup_epochs': (1, 0.0, 5.0),
                'warmup_momentum': (1, 0.0, 0.95),
                'warmup_bias_lr': (1, 0.0, 0.2),
                'box': (1, 0.02, 0.2),
                'cls': (1, 0.2, 4.0),
                'cls_pw': (1, 0.5, 2.0),
                'obj': (1, 0.2, 4.0),
                'obj_pw': (1, 0.5, 2.0),
                'iou_t': (0, 0.1, 0.7),
                'anchor_t': (1, 2.0, 8.0),
                'anchors': (2, 2.0, 10.0),
                'fl_gamma': (0, 0.0, 2.0),
                'hsv_h': (1, 0.0, 0.1),
                'hsv_s': (1, 0.0, 0.9),
                'hsv_v': (1, 0.0, 0.9),
                'degrees': (1, 0.0, 45.0),
                'translate': (1, 0.0, 0.9),
                'scale': (1, 0.0, 0.9),
                'shear': (1, 0.0, 10.0),
                'perspective': (0, 0.0, 0.001),
                'flipud': (1, 0.0, 1.0),
                'fliplr': (0, 0.0, 1.0),
                'mosaic': (1, 0.0, 1.0),
                'mixup': (1, 0.0, 1.0),
                'copy_paste': (1, 0.0, 1.0)}

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
            if 'anchors' not in hyp:
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)

        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')

        for _ in range(opt.evolve):
            if evolve_csv.exists():

                parent = 'single'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min() + 1E-6
                if parent == 'single' or len(x) == 1:

                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()


                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])


            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)


            results = train(hyp.copy(), opt, device, callbacks)


            print_mutation(results, hyp.copy(), save_dir, opt.bucket)


        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):

    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


