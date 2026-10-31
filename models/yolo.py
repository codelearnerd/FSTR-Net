
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg models/repvit_m0_9_t2.yaml
"""

import argparse
import sys, os
from copy import deepcopy
from pathlib import Path

from requests import get
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from models.common import *
from models.experimental import *
from models.lightweight_backbone import (BackboneStage, build_backbone_stages,
                                         restore_backbone_batchnorm)
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)
from torchsummary import summary


try:
    import thop
except ImportError:
    thop = None

import cv2
import numpy as np
import torch.nn.functional as F


class Detect(nn.Module):
    stride = None
    onnx_dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), stride=(), inplace=True,
                 temporal_enabled=False, temporal_seq_len=5,
                 temporal_channels=16, temporal_gain=0.05,
                 temporal_align=True, temporal_max_offset=2.0,
                 temporal_downsample=1):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = inplace
        self.stride = torch.tensor(stride, requires_grad=False)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_adapters = nn.ModuleList(
            CausalPointTemporalAdapter(
                channels,
                state_channels=temporal_channels,
                seq_len=temporal_seq_len,
                residual_gain=temporal_gain,
                align=temporal_align,
                max_offset=temporal_max_offset,
                downsample=temporal_downsample)
            for channels in ch
        ) if self.temporal_enabled else nn.ModuleList()

    def forward(self, x):
        z = []
        temporal_adapters = getattr(self, 'temporal_adapters', ())
        if (getattr(self, 'temporal_enabled', False) and
                len(temporal_adapters) == len(x)):


            x = [
                adapter(feature)
                for adapter, feature in zip(temporal_adapters, x)
            ]


        detect_features = list(x)
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()


            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)


                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]

                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                else:
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y)
        if x and torch.is_tensor(x[0]):
            GLOBAL_MACRO_DETECT_DICT[x[0].device] = {
                'raw': x,
                'anchors': self.anchors,
                'stride': self.stride,
                'features': detect_features,
            }
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z


        return x if self.training else (z, x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])


        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i])\
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class FourierProposal(nn.Module):
    """Formation presence head without learned YOLO rectangle regression.

    It predicts only one objectness logit at every Fourier anchor/grid
    location. Decoded xywh values are deterministic grid/anchor geometry used
    by the existing NMS interface; FTInfer supplies the actual contour center
    and boundary.
    """
    stride = None

    def __init__(self, nc=1, anchors=(), ch=(), stride=(), inplace=True,
                 oracle_geometry=False, oracle_candidate_logit=12.0):
        super().__init__()
        self.nc = int(nc)
        self.no = 6
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer(
            'anchors',
            torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(
            nn.Conv2d(channels, self.na, 1) for channels in ch)
        for conv in self.m:


            nn.init.constant_(conv.bias, -7.0)
        self.inplace = bool(inplace)
        self.stride = torch.tensor(stride, requires_grad=False)
        self.oracle_geometry = bool(oracle_geometry)
        self.oracle_candidate_logit = abs(float(oracle_candidate_logit))

    def _oracle_objectness(self, objectness, level):
        """Build one GT candidate per macro target at its best anchor/scale."""
        if not bool(getattr(self, 'oracle_geometry', False)):
            return None
        oracle = GLOBAL_ORACLE_PERIPHERAL_DICT.get(objectness.device)
        targets = oracle.get('targets') if isinstance(oracle, dict) else None
        if not torch.is_tensor(targets) or targets.numel() == 0:
            return None

        batch, anchors_n, height, width, _ = objectness.shape
        targets = targets.detach().to(
            device=objectness.device, dtype=torch.float32)
        macro = targets[targets[:, 1] == 1.0]
        if macro.numel() == 0:
            return None
        if batch == 1 and (
                macro[:, 0].min() < 0 or macro[:, 0].max() >= batch):
            last_frame = macro[:, 0].max()
            macro = macro[macro[:, 0] == last_frame].clone()
            macro[:, 0] = 0
        else:
            macro = macro[
                (macro[:, 0] >= 0) & (macro[:, 0] < batch)]
        if macro.numel() == 0:
            return None

        candidate_logit = abs(float(getattr(
            self, 'oracle_candidate_logit', 12.0)))
        oracle_logits = objectness.new_full(
            objectness.shape, -candidate_logit)
        image_wh = objectness.new_tensor(
            [width * float(self.stride[level]),
             height * float(self.stride[level])]).float()
        all_anchor_pixels = (
            self.anchors.float() *
            self.stride.to(self.anchors.device).float()[:, None, None]
        ).reshape(-1, 2)
        for target in macro:
            batch_index = int(target[0].item())
            target_wh = (
                target[4:6].to(image_wh.device) * image_wh
            ).clamp_min(1e-4)
            ratio = torch.maximum(
                target_wh[None, :] / all_anchor_pixels,
                all_anchor_pixels / target_wh[None, :])
            best_flat = int(ratio.amax(dim=1).argmin().item())
            best_level = best_flat // self.na
            best_anchor = best_flat % self.na
            if best_level != level:
                continue
            grid_x = min(max(int(target[2].item() * width), 0), width - 1)
            grid_y = min(max(int(target[3].item() * height), 0), height - 1)
            oracle_logits[
                batch_index, best_anchor, grid_y, grid_x, 0
            ] = candidate_logit
        return oracle_logits

    def forward(self, x):
        decoded = []
        proposal_store = {}
        for level in range(self.nl):
            objectness = self.m[level](x[level])
            bs, _, ny, nx = objectness.shape
            objectness = objectness.view(bs, self.na, ny, nx, 1)
            oracle_objectness = self._oracle_objectness(objectness, level)
            if torch.is_tensor(oracle_objectness):
                objectness = oracle_objectness
            proposal_store[(int(ny), int(nx))] = {


                'live_logits': objectness,
                'logits': objectness.detach(),
                'prob': objectness.detach().float().sigmoid(),
                'level': level,
                'stride': float(self.stride[level]),
                'anchor_pixels': (
                    self.anchors[level].detach().float() *
                    float(self.stride[level])),
                'image_hw': (
                    float(ny * self.stride[level]),
                    float(nx * self.stride[level])),
            }
            zeros = objectness.new_zeros((bs, self.na, ny, nx, 4))
            class_logit = objectness.new_full(
                (bs, self.na, ny, nx, 1), 10.0)
            raw = torch.cat((zeros, objectness, class_logit), dim=-1)
            x[level] = raw
            if not self.training:
                if self.grid[level].shape[2:4] != raw.shape[2:4]:
                    self.grid[level], self.anchor_grid[level] =\
                        self._make_grid(nx, ny, level)
                prediction = raw.clone()
                prediction[..., 0:2] = (
                    self.grid[level] + 0.5) * self.stride[level]
                prediction[..., 2:4] = self.anchor_grid[level]
                prediction[..., 4] = prediction[..., 4].sigmoid()
                prediction[..., 5] = 1.0
                decoded.append(prediction)
                proposal_store[(int(ny), int(nx))]['decoded'] = prediction
        if x:
            GLOBAL_FOURIER_PROPOSAL_DICT[x[0].device] = proposal_store
        return x if self.training else (decoded, x)

    def _make_grid(self, nx=20, ny=20, i=0):
        device = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=device),
                 torch.arange(nx, device=device)], indexing='ij')
        else:
            yv, xv = torch.meshgrid(
                [torch.arange(ny, device=device),
                 torch.arange(nx, device=device)])
        grid = torch.stack((xv, yv), 2).expand(
            (1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            self.anchors[i].clone() * self.stride[i]).view(
                1, self.na, 1, 1, 2).expand(
                    1, self.na, ny, nx, 2).float()
        return grid, anchor_grid


class Detect2(Detect):
    stride = None

    def __init__(self, nc=80, anchors=(), ch=(), stride=(), inplace=True):
        super(Detect2, self).__init__(nc, anchors, ch, stride, inplace)
        self.nc = nc
        self.no = 5
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            assert(self.no==1+2+2)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()


            if not self.training:
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 1:3] = y[..., 1:3] * 2 - 1

                    y[..., 3:5] = (y[..., 3:5] * 2) ** 2 * self.anchor_grid[i]


                else:
                    raise Exception("Detect2 forward error")

                z.append(y)
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z

        return x if self.training else (z, x)


class Model(nn.Module):
    def __init__(self, cfg='models/repvit_m0_9_t2.yaml', ch=3, nc=None, anchors=None, ft_coef=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)


        cps_env_overrides = {
            'P2F_CPSI_MAX_GAIN': (
                'ft_point_spectral_max_gain', float),
            'P2F_CPSI_POINT_THRESHOLD': (
                'ft_point_spectral_point_threshold', float),
            'P2F_CPSI_MIN_MASS': (
                'ft_point_spectral_min_mass', float),
            'P2F_CPSI_ROI_SCALE': (
                'ft_point_spectral_roi_scale', float),
        }
        for environment_name, (yaml_name, converter) in\
                cps_env_overrides.items():
            environment_value = os.getenv(environment_name)
            if environment_value not in (None, ''):
                self.yaml[yaml_name] = converter(environment_value)
                LOGGER.info(
                    f'CPSI override: {yaml_name}={self.yaml[yaml_name]}')


        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.ch = ch

        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)
        if ft_coef and ft_coef != self.yaml['ft_coef']:
            LOGGER.info(f"Overriding model.yaml ft_coef={self.yaml['ft_coef']} with ft_coef={ft_coef}")
            self.yaml['ft_coef'] = ft_coef


        self.srange = self.yaml.get('srange', None)
        self.module_idx = {
        }
        self.cms_config = None
        self.mask_dir = [0] * nc
        for i, h in enumerate(self.yaml['head']):
            if h[2] in ['Detect', 'FourierProposal', 'Detect2', 'SubclassInfer', 'FTInfer', 'ValueInfer', 'BioCountInfer']:
                self.module_idx[h[2]] = i + len(self.yaml['backbone'])

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)


        s = 256

        m = self.get_module_byname('Detect')
        if isinstance(m, Detect):
            m.inplace = self.inplace
            self.stride = m.stride
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self._initialize_biases(m)
        proposal = self.get_module_byname('FourierProposal')
        if isinstance(proposal, FourierProposal):
            proposal.inplace = self.inplace
            self.stride = proposal.stride
            check_anchor_order(proposal)
            proposal.anchors /= proposal.stride.view(-1, 1, 1)

        m = self.get_module_byname('Detect2')
        if isinstance(m, Detect2):
            m.inplace = self.inplace
            self.stride = m.stride
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)

        m = self.get_module_byname('FTInfer')
        if m:
            self.stride = m.stride
            m.anchors /= m.stride.view(-1, 1, 1)

        initialize_weights(self)
        restore_backbone_batchnorm(self)
        self.eval()
        self.info()
        self.train()
        LOGGER.info('')


    def set_seq_len(self, seq_len):
        """
        ?Transformer ?
        """
        LOGGER.info(f"Setting sequence length to {seq_len} for temporal modules...")

        for m in self.model:

            if isinstance(m, ASSCAModule):
                m.seq_len = seq_len


        for module in self.modules():
            if module.__class__.__name__ in {
                    'BioCountInfer', 'CausalPointTemporalAdapter'}:
                if hasattr(module, 'seq_len'):
                    module.seq_len = max(int(seq_len), 1)


    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    @torch.no_grad()
    def _fuse_point_candidate_union(self, detect_output, point_output):
        """Fuse dense YOLO candidates with formation evidence from point peaks.

        FTInfer is aligned with Detect at every anchor/grid location.  We
        therefore rescue low-confidence Detect entries in-place, before NMS,
        instead of creating boxes that would have no corresponding Fourier
        coefficients.  This path is deterministic and inference-only; each
        head keeps its native ground-truth objective during training.
        """
        cfg = getattr(self, 'yaml', {})
        if not bool(cfg.get('point_candidate_union', False)):
            return detect_output
        if (not isinstance(detect_output, tuple) or len(detect_output) != 2 or
                not isinstance(point_output, dict)):
            return detect_output
        decoded_levels, raw_levels = detect_output
        point_logits = point_output.get('pred_logits')
        if (not isinstance(decoded_levels, (list, tuple)) or
                not torch.is_tensor(point_logits) or point_logits.ndim != 4):
            return detect_output

        threshold = float(cfg.get('point_candidate_threshold', 0.05))
        kernel = max(int(cfg.get('point_candidate_kernel', 31)), 1)
        kernel = kernel if kernel % 2 else kernel + 1
        min_mass = float(cfg.get('point_candidate_min_mass', 2.0))
        mass_temperature = max(float(cfg.get('point_candidate_mass_temperature', 0.50)), 1e-3)
        gain = float(cfg.get('point_candidate_gain', 0.65))
        class_gain = float(cfg.get('point_candidate_class_gain', 0.25))
        min_objectness = float(cfg.get('point_candidate_min_objectness', 0.001))
        point_topk = max(int(cfg.get('point_candidate_topk', 12)), 0)
        point_min_score = float(cfg.get('point_candidate_min_score', 0.20))
        point_level = int(cfg.get('point_candidate_level', 0))
        point_nms_kernel = max(int(cfg.get('point_candidate_nms_kernel', kernel)), 1)
        point_nms_kernel = point_nms_kernel if point_nms_kernel % 2 else point_nms_kernel + 1

        probability = point_logits.float().sigmoid()


        active = ((probability - threshold) /
                  max(1.0 - threshold, 1e-3)).clamp(0.0, 1.0)
        local_mass = F.avg_pool2d(
            active, kernel_size=kernel, stride=1, padding=kernel // 2) * float(kernel * kernel)
        local_peak = F.max_pool2d(
            probability, kernel_size=kernel, stride=1, padding=kernel // 2)
        presence = torch.sigmoid((local_mass - min_mass) / mass_temperature)
        support_map = (local_peak * presence).clamp(0.0, 1.0)


        point_proposals = [[] for _ in range(support_map.shape[0])]
        suppress_radius = point_nms_kernel // 2
        for batch_idx in range(support_map.shape[0]):
            work = support_map[batch_idx, 0].clone()
            for _ in range(point_topk):
                score, flat_idx = work.reshape(-1).max(dim=0)
                if float(score) < point_min_score:
                    break
                py = int(flat_idx // work.shape[1])
                px = int(flat_idx % work.shape[1])
                point_proposals[batch_idx].append((px, py, score))
                y0, y1 = max(py - suppress_radius, 0), min(py + suppress_radius + 1, work.shape[0])
                x0, x1 = max(px - suppress_radius, 0), min(px + suppress_radius + 1, work.shape[1])
                work[y0:y1, x0:x1] = 0.0

        detect_module = self.get_module_byname('Detect')
        strides = getattr(detect_module, 'stride', None)
        if not torch.is_tensor(strides):
            return detect_output

        fused_levels = []
        sampled_means = []
        for level_idx, level in enumerate(decoded_levels):
            if (not torch.is_tensor(level) or level.ndim != 5 or
                    level.shape[0] != support_map.shape[0] or level.shape[-1] < 5):
                fused_levels.append(level)
                continue
            batch, anchors_n, height, width, _ = level.shape
            stride = float(strides[level_idx])
            image_h, image_w = height * stride, width * stride
            centers = level[..., :2].float()
            grid_x = centers[..., 0] * (2.0 / max(image_w - 1.0, 1.0)) - 1.0
            grid_y = centers[..., 1] * (2.0 / max(image_h - 1.0, 1.0)) - 1.0
            sample_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
                batch, anchors_n * height, width, 2)
            sampled = F.grid_sample(
                support_map, sample_grid, mode='bilinear',
                padding_mode='zeros', align_corners=True).reshape(
                    batch, anchors_n, height, width)
            fused = level.clone()
            objectness = fused[..., 4].float().clamp(0.0, 1.0)


            yolo_evidence = (objectness >= min_objectness).to(sampled.dtype)
            rescue = (gain * sampled * yolo_evidence).clamp(0.0, 1.0)


            if level_idx == point_level and point_topk:
                point_rescue = torch.zeros_like(rescue)
                class_score = (fused[..., 5:].float().amax(dim=-1)
                               if fused.shape[-1] > 5 else torch.ones_like(objectness))
                anchor_score = objectness * class_score
                map_h, map_w = support_map.shape[-2:]
                for batch_idx, proposals in enumerate(point_proposals):
                    for px, py, proposal_score in proposals:
                        gx = min(int((px + 0.5) * width / map_w), width - 1)
                        gy = min(int((py + 0.5) * height / map_h), height - 1)
                        anchor_idx = int(anchor_score[batch_idx, :, gy, gx].argmax())
                        point_rescue[batch_idx, anchor_idx, gy, gx] = gain * proposal_score
                rescue = torch.maximum(rescue, point_rescue.clamp(0.0, 1.0))
            fused[..., 4] = (1.0 - (1.0 - objectness) * (1.0 - rescue)).to(fused.dtype)
            if fused.shape[-1] > 5 and class_gain > 0.0:


                rescue_support = (rescue / max(gain, 1e-6)).clamp(0.0, 1.0)
                class_rescue = (class_gain * rescue_support).clamp(0.0, 1.0).unsqueeze(-1)
                classes = fused[..., 5:].float().clamp(0.0, 1.0)
                fused[..., 5:] = (
                    1.0 - (1.0 - classes) * (1.0 - class_rescue)).to(fused.dtype)
            fused_levels.append(fused)
            sampled_means.append(sampled.mean())

        self.point_candidate_union_stats = {
            'support_max': support_map.max().detach(),
            'support_mean': support_map.mean().detach(),
            'sampled_mean': (torch.stack(sampled_means).mean().detach()
                             if sampled_means else support_map.new_zeros(())),
        }
        return fused_levels, raw_levels

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]

            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _forward_once(self, x, profile=False, visualize=False):

        GLOBAL_MACRO_DETECT_DICT.pop(x.device, None)
        GLOBAL_POINT_DICT.pop(x.device, None)
        GLOBAL_POINT_FEATURE_DICT.pop(x.device, None)
        GLOBAL_POINT_MOTION_DICT.pop(x.device, None)
        GLOBAL_FORMATION_SUPPORT_DICT.pop(x.device, None)
        GLOBAL_FOURIER_PROPOSAL_DICT.pop(x.device, None)
        GLOBAL_CANDIDATE_GEOMETRY_DICT.pop(x.device, None)
        GLOBAL_ORACLE_FOURIER_BASE_DICT.pop(x.device, None)
        y, dt = [], []
        out = []
        p2p_output = None
        guidance_map_output = None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)


            module_out = m(x)
            if self.training and isinstance(module_out, tuple) and len(module_out) == 2:
                x = module_out[0]
                guidance_map_output = module_out[1]
            else:
                x = module_out
            y.append(x if m.i in self.save else None)

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

            if type(m).__name__ in ['Detect', 'FourierProposal', 'Detect2', 'SubclassInfer', 'FTInfer', 'ValueInfer', 'P2PInfer', 'BioCountInfer']:
                if type(m).__name__ in ['P2PInfer', 'BioCountInfer']:
                    p2p_output = x
                else:
                    if self.training:
                        out.extend(x)
                    else:
                        out.append(x)


        if self.training and guidance_map_output is not None:
            self.guidance_map = guidance_map_output
        if (not self.training and p2p_output is not None and out):
            out[0] = self._fuse_point_candidate_union(out[0], p2p_output)
        if p2p_output is not None:
            out.append(p2p_output)

        return out


    def _descale_pred(self, p, flips, scale, img_size):

        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):

        detection_head = self.get_module_byname('Detect')
        detection_head = detection_head if detection_head is not None else self.get_module_byname('FourierProposal')
        nl = detection_head.nl
        g = sum(4 ** x for x in range(nl))
        e = 1
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))
        y[0] = y[0][:, :-i]
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))
        y[-1] = y[-1][:, i:]
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, (Detect, FourierProposal))
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, m, cf=None):


        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.self.get_module_byname('Detect')
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        self.info()
        return self

    def autoshape(self):
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        return m

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):

        self = super()._apply(fn)
        m = self.get_module_byname('Detect')
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        proposal = self.get_module_byname('FourierProposal')
        if isinstance(proposal, FourierProposal):
            proposal.stride = fn(proposal.stride)
            proposal.grid = list(map(fn, proposal.grid))
            if isinstance(proposal.anchor_grid, list):
                proposal.anchor_grid = list(map(fn, proposal.anchor_grid))
        return self

    def get_module_byname(self, name:str):
        if not hasattr(self, 'module_idx'):
            self.module_idx = {
            }
            for i, h in enumerate(self.yaml['head']):
                if h[2] in ['Detect', 'FourierProposal', 'Detect2', 'SubclassInfer', 'ValueInfer', 'FTInfer', 'BioCountInfer']:
                    self.module_idx[h[2]] = i + len(self.yaml['backbone'])
        idx = self.module_idx.get(name, -1)
        return self.model[idx] if idx != -1 else None

    def reset_bn_stats(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()


def parse_model(d, ch):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    anchorsab = d.get('anchorsab', None)

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    save2 = set()

    strides = []
    backbone_stages = None
    if any(layer[2] == 'BackboneStage' for layer in d['backbone']):
        backbone_stages = build_backbone_stages(
            pretrained=bool(d.get('backbone_pretrained', True)))
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        now_stride = strides[f[0] if isinstance(f, list)else f] if len(strides)>0 else 1
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                if isinstance(a, str):


                    args[j] = d[a] if a in d else eval(a)
                else:
                    args[j] = a
            except (NameError, KeyError):
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, DetachedConv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, CBAM, CSPC, CSPCAM, CSPSAM, C3_ASSCA, SPPF_Transformer]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [Focus, Conv, DetachedConv, GhostConv, DWConv, GhostBottleneck, MixConv2d, CrossConv]:
                now_stride = now_stride * args[3]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, CSPC,CSPCAM, CSPSAM, C3_ASSCA]:
                args.insert(2, n)
                n = 1
        elif m is BackboneStage:
            stage = int(args[0])
            if n != 1 or not 0 <= stage < 5:
                raise ValueError('BackboneStage requires n=1 and stage in 0..4')
            if ch[f] != BackboneStage.input_channels[stage]:
                raise ValueError(f'Invalid backbone stage {stage} input: {ch[f]} channels')
            c2 = BackboneStage.output_channels[stage]
            now_stride *= 2
        elif m is ASSCAModule:
            c1 = ch[f]
            c2 = c1


        elif m in [PointFeatureGate, PeripheralPointFourierFusion,
                   OutwardPointFourierFusion,
                   CandidateDirectionalFourierBridge,
                   YoloPointFourierFeatureBridge,
                   FourierPointHeatmapFusion, FormationSupportBridge,
                   OraclePeripheralFourierBridge]:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is GlobalClusterBlock:
            c1 = ch[f]
            c2 = args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
        elif m is nn.Upsample:
            now_stride = int(now_stride / args[1])
            c2 = ch[f]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m in [Concat, ConcatDetachSkip]:
            c2 = sum(ch[x] for x in f)
        elif m in [Detect, FourierProposal, Detect2, ValueInfer, SubclassInfer, FTInfer, P2PInfer, BioCountInfer]:
            bio_detach_features = None
            ft_support_residual = False
            if m is BioCountInfer and len(args) > 1:
                bio_detach_features = bool(args.pop(1))
            if m is FTInfer and len(args) > 4:
                ft_support_residual = bool(args.pop(4))
            args.append([ch[x] for x in f])
            if isinstance(args[1], int) and m not in [ValueInfer, SubclassInfer, FTInfer, P2PInfer, BioCountInfer]:
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append([strides[x] for x in f])
            if bio_detach_features is not None:
                args.append(bio_detach_features)
                args.extend([
                    bool(d.get('point_yolo_prior', False)),
                    float(d.get('point_yolo_box_scale', 1.35)),
                    float(d.get('point_yolo_epsilon', 0.20)),
                    float(d.get('point_yolo_conf', 0.05)),
                    int(d.get('point_yolo_topk', 20)),
                    bool(d.get('point_contour_prior', True)),
                    float(d.get('point_yolo_gain', 0.50)),
                    bool(d.get('point_temporal_enabled', False)),
                    int(d.get('point_temporal_seq_len', 5)),
                    int(d.get('point_temporal_channels', 32)),
                    float(d.get('point_temporal_gain', 0.05)),
                    bool(d.get('point_temporal_align', True)),
                    float(d.get('point_temporal_max_offset', 2.0)),
                    int(d.get('point_out_scale', 2)),
                    int(d.get('point_temporal_downsample', 1)),
                    float(d.get('point_temporal_suppress_gain', 0.0)),
                    float(d.get('point_temporal_contrast_gain', 0.0)),
                    float(d.get('point_yolo_outside_floor', 1.0)),
                    float(d.get('point_yolo_logit_gain', 0.0)),
                    float(d.get('point_yolo_softness', 1.5)),
                    float(d.get('point_yolo_nms_iou', 1.0)),
                    str(d.get('point_temporal_mode', 'gru')),
                    float(d.get('point_motion_spatial_floor', 0.15)),
                    float(d.get('point_motion_enhance_gain', 0.15)),
                    float(d.get('point_motion_threshold', 2.0)),
                    float(d.get('point_motion_temperature', 0.5)),
                    int(d.get('point_motion_track_radius', 2)),
                    bool(d.get('point_motion_output_gate', False)),
                    float(d.get('point_motion_output_floor', 0.02)),
                    float(d.get('point_motion_output_gamma', 2.0)),
                    bool(d.get('point_candidate_recall_enabled', False)),
                    float(d.get('point_candidate_recall_conf', 0.05)),
                    int(d.get('point_candidate_recall_topk', 4)),
                    float(d.get('point_candidate_recall_anchor_scale', 1.5)),
                    float(d.get('point_candidate_recall_min_pixels', 40.0)),
                    float(d.get('point_candidate_recall_max_pixels', 96.0)),
                    float(d.get('point_candidate_recall_softness', 0.20)),
                    float(d.get('point_candidate_recall_max_gain', 0.25)),
                    float(d.get('point_candidate_recall_gain_init', -1.0)),
                    bool(d.get('fourier_to_point_enabled', False)),
                    bool(d.get('fourier_to_point_detach', True)),
                    float(d.get('fourier_to_point_scale', 1.60)),
                    float(d.get('fourier_to_point_margin_pixels', 6.0)),
                    float(d.get('fourier_to_point_outside_floor', 0.20)),
                    float(d.get('fourier_to_point_gamma', 1.0)),
                    float(d.get('fourier_to_point_candidate_conf', 0.20)),
                    int(d.get('fourier_to_point_topk', 4)),
                    float(d.get('fourier_to_point_softness', 0.20)),
                    bool(d.get('fourier_to_point_fallback_identity', True)),
                    bool(d.get('fourier_to_point_core_identity', False)),
                    float(d.get('fourier_to_point_core_threshold', 0.25)),
                    bool(d.get('fourier_to_point_density_rescue', False)),
                    int(d.get('fourier_to_point_density_kernel', 7)),
                    float(d.get('fourier_to_point_density_min_mass', 1.5)),
                    float(d.get('fourier_to_point_density_temperature', 0.25)),
                    float(d.get('fourier_to_point_density_point_threshold', 0.10)),
                    float(d.get('fourier_to_point_rescue_max_distance', 16.0)),
                    float(d.get('fourier_to_point_rescue_gain', 0.70)),
                ])
            if m is Detect:
                args.extend([
                    True,
                    bool(d.get('detect_temporal_enabled', False)),
                    int(d.get('detect_temporal_seq_len', 5)),
                    int(d.get('detect_temporal_channels', 16)),
                    float(d.get('detect_temporal_gain', 0.05)),
                    bool(d.get('detect_temporal_align', True)),
                    float(d.get('detect_temporal_max_offset', 2.0)),
                    int(d.get('detect_temporal_downsample', 1)),
                ])
            if m is FourierProposal:
                args.extend([
                    True,
                    bool(d.get('ft_geometry_oracle', False)),
                    float(d.get('ft_geometry_oracle_candidate_logit', 12.0)),
                ])
            if m is FTInfer:
                args.extend([
                    ft_support_residual,
                    int(d.get('ft_geometry_directions', 32)),
                    int(d.get('ft_geometry_radial_samples', 20)),
                    float(d.get('ft_geometry_support_threshold', 0.15)),
                    float(d.get('ft_geometry_temperature', 0.20)),
                    float(d.get('ft_geometry_outward_bias', 5.0)),
                    float(d.get('ft_geometry_radius_scale', 1.25)),
                    float(d.get('ft_geometry_presence_gain', 4.0)),
                    int(d.get('ft_geometry_chunk_size', 1024)),
                    float(d.get('ft_geometry_candidate_threshold', 0.05)),
                    float(d.get(
                        'ft_geometry_candidate_gate_threshold', 0.20)),
                    float(d.get(
                        'ft_geometry_candidate_gate_temperature', 0.05)),
                    int(d.get('ft_geometry_candidate_topk', 16)),
                    float(d.get('ft_geometry_residual_gain', 0.75)),
                    int(d.get('ft_geometry_debug_topk', 4)),
                    float(d.get('ft_geometry_center_blend', 1.0)),
                    int(d.get('ft_geometry_support_mass_kernel', 1)),
                    float(d.get(
                        'ft_geometry_support_mass_threshold', 0.0)),
                    float(d.get(
                        'ft_geometry_support_mass_temperature', 0.15)),
                    int(d.get('ft_geometry_angular_smooth_kernel', 1)),
                    float(d.get(
                        'ft_geometry_angular_spike_gain', 2.0)),
                    float(d.get(
                        'ft_geometry_expansion_cap_ratio', 0.0)),
                    str(d.get('ft_geometry_point_mode', 'soft')),
                    float(d.get('ft_geometry_peak_threshold', 0.60)),
                    int(d.get('ft_geometry_peak_nms_kernel', 3)),
                    int(d.get('ft_geometry_peak_topk', 100)),
                    float(d.get(
                        'ft_geometry_candidate_roi_scale', 2.75)),
                    float(d.get(
                        'ft_geometry_candidate_roi_min_pixels', 40.0)),
                    float(d.get(
                        'ft_geometry_candidate_roi_max_pixels', 96.0)),
                    float(d.get(
                        'ft_geometry_peak_padding_pixels', 3.5)),
                    float(d.get(
                        'ft_geometry_discrete_center_blend', 0.70)),
                    int(d.get('ft_geometry_peak_min_count', 2)),
                    float(d.get(
                        'ft_geometry_peak_outlier_mad_gain', 3.5)),
                    bool(d.get('ft_geometry_oracle', False)),
                    float(d.get('ft_geometry_oracle_point_sigma', 0.75)),
                    float(d.get(
                        'ft_geometry_oracle_candidate_logit', 12.0)),
                    bool(d.get('ft_temporal_enabled', False)),
                    int(d.get('ft_temporal_seq_len', 5)),
                    int(d.get('ft_temporal_channels', 16)),
                    float(d.get('ft_temporal_gain', 0.05)),
                    bool(d.get('ft_temporal_align', True)),
                    float(d.get('ft_temporal_max_offset', 2.0)),
                    int(d.get('ft_temporal_downsample', 1)),
                    float(d.get('ft_temporal_suppress_gain', 0.0)),
                    float(d.get('ft_temporal_contrast_gain', 0.0)),
                    bool(d.get('ft_point_spectral_enabled', False)),
                    int(d.get('ft_point_spectral_candidate_topk', 12)),
                    int(d.get('ft_point_spectral_directions', 64)),
                    int(d.get('ft_point_spectral_radial_samples', 24)),
                    float(d.get(
                        'ft_point_spectral_radial_softmax_beta', 0.0)),
                    float(d.get('ft_point_spectral_roi_scale', 2.25)),
                    float(d.get(
                        'ft_point_spectral_roi_min_pixels', 32.0)),
                    float(d.get(
                        'ft_point_spectral_roi_max_pixels', 96.0)),
                    float(d.get(
                        'ft_point_spectral_assignment_temperature', 0.12)),
                    float(d.get(
                        'ft_point_spectral_direction_temperature', 0.10)),
                    float(d.get(
                        'ft_point_spectral_point_threshold', 0.20)),
                    float(d.get(
                        'ft_point_spectral_point_temperature', 0.08)),
                    float(d.get(
                        'ft_point_spectral_point_grad_scale', 1.0)),
                    float(d.get('ft_point_spectral_min_mass', 0.75)),
                    float(d.get(
                        'ft_point_spectral_mass_temperature', 0.25)),
                    float(d.get('ft_point_spectral_max_gain', 0.75)),
                    float(d.get('ft_point_spectral_gate_init', -1.50)),
                    d.get('ft_point_spectral_gate_floor', None),
                    str(d.get(
                        'ft_point_spectral_fusion_mode', 'coefficient_blend')),
                    float(d.get('ft_point_spectral_order_decay', 0.04)),
                    float(d.get(
                        'ft_point_spectral_candidate_threshold', 0.05)),
                    float(d.get(
                        'ft_point_spectral_candidate_temperature', 0.08)),
                    float(d.get('ft_point_spectral_padding_pixels', 2.0)),
                    float(d.get(
                        'ft_point_spectral_expansion_cap_ratio', 1.50)),
                    float(d.get(
                        'ft_point_spectral_min_cap_roi_ratio', 0.0)),
                    int(d.get('ft_point_spectral_debug_topk', 4)),
                ])

            c2 = args[0] if m not in [Detect, FourierProposal, Detect2, FTInfer] else ch[f[0]]

        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        strides.append(now_stride)

        m_ = (backbone_stages[int(args[0])] if m is BackboneStage else
              nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args))
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')

        for x in ([f] if isinstance(f, int) else f):
            if x != -1:
                save2.add(x % i)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    save = list(save2)
    return nn.Sequential(*layers), sorted(save)


class ValueInfer(nn.Module):
    def __init__(self, nv, anchors=(), ch=(), stride=()):
        '''
        na: nchor?        nv: ?
        '''
        super().__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.nv = nv
        self.m = nn.ModuleList(nn.Conv2d(x, self.nv * self.na , 1) for x in ch)
        self.stride = torch.tensor(stride, requires_grad=False)

    def forward(self, x):
        z = []
        for i in range(self.na):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.nv, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                z.append(x[i].sigmoid())
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)


class SubclassInfer(nn.Module):
    def __init__(self, ns=1, anchors=(), ch=()):
        '''
        na: nchor?        ns: ?
        '''
        super().__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.ns = ns
        self.m = nn.ModuleList(nn.Conv2d(x, self.ns * self.na , 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.ns, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                z.append(x[i].sigmoid())
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)


class FTInfer(nn.Module):
    stride=None
    def __init__(self, ft_coef, lamda, d0, anchors=(), ch=(), stride=(),
                 support_residual=False, geometry_directions=32,
                 geometry_radial_samples=20, geometry_support_threshold=0.15,
                 geometry_temperature=0.20, geometry_outward_bias=5.0,
                 geometry_radius_scale=1.25, geometry_presence_gain=4.0,
                 geometry_chunk_size=1024,
                 geometry_candidate_threshold=0.05,
                 geometry_candidate_gate_threshold=0.20,
                 geometry_candidate_gate_temperature=0.05,
                 geometry_candidate_topk=16,
                 geometry_residual_gain=0.75,
                 geometry_debug_topk=4,
                 geometry_center_blend=1.0,
                 geometry_support_mass_kernel=1,
                 geometry_support_mass_threshold=0.0,
                 geometry_support_mass_temperature=0.15,
                 geometry_angular_smooth_kernel=1,
                 geometry_angular_spike_gain=2.0,
                 geometry_expansion_cap_ratio=0.0,
                 geometry_point_mode='soft',
                 geometry_peak_threshold=0.60,
                 geometry_peak_nms_kernel=3,
                 geometry_peak_topk=100,
                 geometry_candidate_roi_scale=2.75,
                 geometry_candidate_roi_min_pixels=40.0,
                 geometry_candidate_roi_max_pixels=96.0,
                 geometry_peak_padding_pixels=3.5,
                 geometry_discrete_center_blend=0.70,
                 geometry_peak_min_count=2,
                 geometry_peak_outlier_mad_gain=3.5,
                 geometry_oracle=False,
                 geometry_oracle_point_sigma=0.75,
                 geometry_oracle_candidate_logit=12.0,
                 temporal_enabled=False,
                 temporal_seq_len=5, temporal_channels=16,
                 temporal_gain=0.05, temporal_align=True,
                 temporal_max_offset=2.0, temporal_downsample=1,
                 temporal_suppress_gain=0.0,
                 temporal_contrast_gain=0.0,
                 point_spectral_enabled=False,
                 point_spectral_candidate_topk=12,
                 point_spectral_directions=64,
                 point_spectral_radial_samples=24,
                 point_spectral_radial_softmax_beta=0.0,
                 point_spectral_roi_scale=2.25,
                 point_spectral_roi_min_pixels=32.0,
                 point_spectral_roi_max_pixels=96.0,
                 point_spectral_assignment_temperature=0.12,
                 point_spectral_direction_temperature=0.10,
                 point_spectral_point_threshold=0.20,
                 point_spectral_point_temperature=0.08,
                 point_spectral_point_grad_scale=1.0,
                 point_spectral_min_mass=0.75,
                 point_spectral_mass_temperature=0.25,
                 point_spectral_max_gain=0.75,
                 point_spectral_gate_init=-1.50,
                 point_spectral_gate_floor=None,
                 point_spectral_fusion_mode='coefficient_blend',
                 point_spectral_order_decay=0.04,
                 point_spectral_candidate_threshold=0.05,
                 point_spectral_candidate_temperature=0.08,
                 point_spectral_padding_pixels=2.0,
                 point_spectral_expansion_cap_ratio=1.50,
                 point_spectral_min_cap_roi_ratio=0.0,
                 point_spectral_debug_topk=4,
                 inplace=True, ec=0.3):
        '''
        na: nchor?        ft_coef: ?coef?        '''
        super().__init__()
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.ft_coef = ft_coef
        self.ft_length = 2 + 4 * ft_coef
        self.m = nn.ModuleList(nn.Conv2d(x, self.ft_length * self.na , 1) for x in ch)
        self.support_residual = bool(support_residual)
        if self.support_residual:


            self.geometry_directions = max(int(geometry_directions), 2 * ft_coef + 1)
            self.geometry_radial_samples = max(int(geometry_radial_samples), 4)
            self.geometry_support_threshold = float(geometry_support_threshold)
            self.geometry_temperature = max(float(geometry_temperature), 1e-3)
            self.geometry_outward_bias = float(geometry_outward_bias)
            self.geometry_radius_scale = max(float(geometry_radius_scale), 0.1)
            self.geometry_presence_gain = max(float(geometry_presence_gain), 0.1)
            self.geometry_chunk_size = max(int(geometry_chunk_size), 1)
            self.geometry_candidate_threshold = max(
                float(geometry_candidate_threshold), 0.0)
            self.geometry_candidate_gate_threshold = float(
                geometry_candidate_gate_threshold)
            self.geometry_candidate_gate_temperature = max(
                float(geometry_candidate_gate_temperature), 1e-3)
            self.geometry_candidate_topk = max(
                int(geometry_candidate_topk), 1)
            self.geometry_residual_gain = max(
                float(geometry_residual_gain), 0.0)
            self.geometry_debug_topk = max(int(geometry_debug_topk), 1)
            self.geometry_center_blend = min(max(
                float(geometry_center_blend), 0.0), 1.0)
            support_mass_kernel = max(
                int(geometry_support_mass_kernel), 1)
            self.geometry_support_mass_kernel = (
                support_mass_kernel if support_mass_kernel % 2 else
                support_mass_kernel + 1)
            self.geometry_support_mass_threshold = max(
                float(geometry_support_mass_threshold), 0.0)
            self.geometry_support_mass_temperature = max(
                float(geometry_support_mass_temperature), 1e-3)
            angular_kernel = max(
                int(geometry_angular_smooth_kernel), 1)
            self.geometry_angular_smooth_kernel = (
                angular_kernel if angular_kernel % 2 else
                angular_kernel + 1)
            self.geometry_angular_spike_gain = max(
                float(geometry_angular_spike_gain), 1.0)
            self.geometry_expansion_cap_ratio = max(
                float(geometry_expansion_cap_ratio), 0.0)
            point_mode = str(geometry_point_mode).strip().lower()
            self.geometry_point_mode = (
                point_mode if point_mode in {'soft', 'discrete'} else 'soft')
            self.geometry_peak_threshold = min(max(
                float(geometry_peak_threshold), 0.0), 1.0)
            peak_kernel = max(int(geometry_peak_nms_kernel), 1)
            self.geometry_peak_nms_kernel = (
                peak_kernel if peak_kernel % 2 else peak_kernel + 1)
            self.geometry_peak_topk = max(
                int(geometry_peak_topk), 1)
            self.geometry_candidate_roi_scale = max(
                float(geometry_candidate_roi_scale), 1.0)
            self.geometry_candidate_roi_min_pixels = max(
                float(geometry_candidate_roi_min_pixels), 0.0)
            self.geometry_candidate_roi_max_pixels = max(
                float(geometry_candidate_roi_max_pixels),
                self.geometry_candidate_roi_min_pixels)
            self.geometry_peak_padding_pixels = max(
                float(geometry_peak_padding_pixels), 0.0)
            self.geometry_discrete_center_blend = min(max(
                float(geometry_discrete_center_blend), 0.0), 1.0)
            self.geometry_peak_min_count = max(
                int(geometry_peak_min_count), 1)
            self.geometry_peak_outlier_mad_gain = max(
                float(geometry_peak_outlier_mad_gain), 0.0)
            self.geometry_oracle = bool(geometry_oracle)
            self.geometry_oracle_point_sigma = max(
                float(geometry_oracle_point_sigma), 0.1)
            self.geometry_oracle_candidate_logit = abs(
                float(geometry_oracle_candidate_logit))
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.inplace = inplace
        self.lamda = lamda
        self.d0 = d0
        self.cm = None
        self.ec = ec
        self.stride = torch.tensor(stride, requires_grad=False)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_adapters = nn.ModuleList(
            CausalPointTemporalAdapter(
                channels, state_channels=temporal_channels,
                seq_len=temporal_seq_len, residual_gain=temporal_gain,
                align=temporal_align, max_offset=temporal_max_offset,
                downsample=temporal_downsample,
                suppression_gain=temporal_suppress_gain,
                contrast_gain=temporal_contrast_gain)
            for channels in ch
        ) if self.temporal_enabled else nn.ModuleList()
        self.point_spectral_enabled = bool(point_spectral_enabled)
        self.point_spectral_candidate_topk = max(
            int(point_spectral_candidate_topk), 1)
        self.point_spectral_directions = max(
            int(point_spectral_directions), 2 * self.ft_coef + 1)
        self.point_spectral_radial_samples = max(
            int(point_spectral_radial_samples), 8)
        self.point_spectral_radial_softmax_beta = max(
            float(point_spectral_radial_softmax_beta), 0.0)
        self.point_spectral_roi_scale = max(
            float(point_spectral_roi_scale), 1.0)
        self.point_spectral_roi_min_pixels = max(
            float(point_spectral_roi_min_pixels), 0.0)
        self.point_spectral_roi_max_pixels = max(
            float(point_spectral_roi_max_pixels),
            self.point_spectral_roi_min_pixels)
        self.point_spectral_assignment_temperature = max(
            float(point_spectral_assignment_temperature), 1e-3)
        self.point_spectral_direction_temperature = max(
            float(point_spectral_direction_temperature), 1e-3)
        self.point_spectral_point_threshold = min(max(
            float(point_spectral_point_threshold), 0.0), 1.0)
        self.point_spectral_point_temperature = max(
            float(point_spectral_point_temperature), 1e-3)
        self.point_spectral_point_grad_scale = min(max(
            float(point_spectral_point_grad_scale), 0.0), 1.0)
        self.point_spectral_min_mass = max(
            float(point_spectral_min_mass), 0.0)
        self.point_spectral_mass_temperature = max(
            float(point_spectral_mass_temperature), 1e-3)
        self.point_spectral_max_gain = min(max(
            float(point_spectral_max_gain), 0.0), 1.0)
        self.point_spectral_order_decay = max(
            float(point_spectral_order_decay), 0.0)
        if point_spectral_gate_floor is None:
            gate_floor = torch.zeros(self.ft_coef, dtype=torch.float32)
        else:
            gate_floor = torch.as_tensor(
                point_spectral_gate_floor, dtype=torch.float32).flatten()
            if gate_floor.numel() == 1:
                gate_floor = gate_floor.repeat(self.ft_coef)
            elif gate_floor.numel() < self.ft_coef:
                gate_floor = F.pad(
                    gate_floor, (0, self.ft_coef - gate_floor.numel()))
            gate_floor = gate_floor[:self.ft_coef]
        self.register_buffer(
            'point_spectral_gate_floor', gate_floor.clamp(0.0, 1.0),
            persistent=True)
        self.point_spectral_fusion_mode = str(
            point_spectral_fusion_mode).strip().lower()
        if self.point_spectral_fusion_mode not in {
                'coefficient_blend', 'phase_aligned_radial_union'}:
            raise ValueError(
                'point_spectral_fusion_mode must be coefficient_blend or '
                'phase_aligned_radial_union, got '
                f'{self.point_spectral_fusion_mode!r}')
        self.point_spectral_candidate_threshold = min(max(
            float(point_spectral_candidate_threshold), 0.0), 1.0)
        self.point_spectral_candidate_temperature = max(
            float(point_spectral_candidate_temperature), 1e-3)
        self.point_spectral_padding_pixels = max(
            float(point_spectral_padding_pixels), 0.0)
        self.point_spectral_expansion_cap_ratio = max(
            float(point_spectral_expansion_cap_ratio), 0.0)
        self.point_spectral_min_cap_roi_ratio = max(
            float(point_spectral_min_cap_roi_ratio), 0.0)
        self.point_spectral_debug_topk = max(
            int(point_spectral_debug_topk), 1)
        if self.point_spectral_enabled:


            self.point_spectral_gate_heads = nn.ModuleList(
                nn.Conv2d(channels, self.na * self.ft_coef, 1)
                for channels in ch)
            for gate_head in self.point_spectral_gate_heads:
                nn.init.zeros_(gate_head.weight)
                nn.init.constant_(
                    gate_head.bias, float(point_spectral_gate_init))
        else:
            self.point_spectral_gate_heads = nn.ModuleList()

    def _geometric_settings(self):
        return (
            max(int(getattr(self, 'geometry_directions', 32)), 2 * self.ft_coef + 1),
            max(int(getattr(self, 'geometry_radial_samples', 20)), 4),
            float(getattr(self, 'geometry_support_threshold', 0.15)),
            max(float(getattr(self, 'geometry_temperature', 0.20)), 1e-3),
            float(getattr(self, 'geometry_outward_bias', 5.0)),
            max(float(getattr(self, 'geometry_radius_scale', 1.25)), 0.1),
            max(float(getattr(self, 'geometry_presence_gain', 4.0)), 0.1),
            max(int(getattr(self, 'geometry_chunk_size', 1024)), 1),
        )

    def _candidate_geometric_settings(self):
        return (
            max(float(getattr(
                self, 'geometry_candidate_threshold', 0.05)), 0.0),
            float(getattr(
                self, 'geometry_candidate_gate_threshold', 0.20)),
            max(float(getattr(
                self, 'geometry_candidate_gate_temperature', 0.05)), 1e-3),
            max(int(getattr(self, 'geometry_candidate_topk', 16)), 1),
            max(float(getattr(self, 'geometry_residual_gain', 0.75)), 0.0),
            max(int(getattr(self, 'geometry_debug_topk', 4)), 1),
        )

    def _discrete_point_settings(self):
        """Return deployable predicted-peak geometry settings.

        The discrete path intentionally consumes the same local maxima that
        the point detector exposes at inference instead of treating every
        low-confidence heatmap pixel as geometric support.
        """
        peak_kernel = max(int(getattr(
            self, 'geometry_peak_nms_kernel', 3)), 1)
        if not peak_kernel % 2:
            peak_kernel += 1
        roi_min = max(float(getattr(
            self, 'geometry_candidate_roi_min_pixels', 40.0)), 0.0)
        roi_max = max(float(getattr(
            self, 'geometry_candidate_roi_max_pixels', 96.0)), roi_min)
        return (
            str(getattr(
                self, 'geometry_point_mode', 'soft')).strip().lower(),
            min(max(float(getattr(
                self, 'geometry_peak_threshold', 0.60)), 0.0), 1.0),
            peak_kernel,
            max(int(getattr(self, 'geometry_peak_topk', 100)), 1),
            max(float(getattr(
                self, 'geometry_candidate_roi_scale', 2.75)), 1.0),
            roi_min,
            roi_max,
            max(float(getattr(
                self, 'geometry_peak_padding_pixels', 3.5)), 0.0),
            min(max(float(getattr(
                self, 'geometry_discrete_center_blend', 0.70)),
                0.0), 1.0),
            max(int(getattr(
                self, 'geometry_peak_min_count', 2)), 1),
            max(float(getattr(
                self, 'geometry_peak_outlier_mad_gain', 3.5)), 0.0),
        )

    @staticmethod
    def _discrete_peak_support(
            support, reliability, threshold, nms_kernel):
        """Extract a sparse confidence map of reliable local point maxima."""
        source = support.detach().float().clamp(0.0, 1.0)
        reliable = (
            reliability.detach().float() >= 0.5
            if torch.is_tensor(reliability) else
            torch.ones_like(source, dtype=torch.bool))


        ranking = source + 1e-3 * F.avg_pool2d(
            source, 3, stride=1, padding=1)
        local_maximum = F.max_pool2d(
            ranking, nms_kernel, stride=1, padding=nms_kernel // 2)
        peak_mask = (
            reliable &
            (source >= float(threshold)) &
            (ranking >= local_maximum - 1e-7))
        return source * peak_mask.to(dtype=source.dtype)

    @torch.no_grad()
    def _oracle_geometry_level(self, raw, level):
        """Create exact macro candidates, centers and micro-point support."""
        if not bool(getattr(self, 'geometry_oracle', False)):
            return None
        oracle = GLOBAL_ORACLE_PERIPHERAL_DICT.get(raw.device)
        targets = oracle.get('targets') if isinstance(oracle, dict) else None
        if not torch.is_tensor(targets) or targets.numel() == 0:
            return None

        batch, anchors_n, height, width, _ = raw.shape
        targets = targets.detach().to(device=raw.device, dtype=torch.float32)
        if batch == 1 and (
                targets[:, 0].min() < 0 or targets[:, 0].max() >= batch):
            last_frame = targets[:, 0].max()
            targets = targets[targets[:, 0] == last_frame].clone()
            targets[:, 0] = 0
        else:
            targets = targets[
                (targets[:, 0] >= 0) & (targets[:, 0] < batch)]
        if targets.numel() == 0:
            return None

        macro = targets[targets[:, 1] == 1.0]
        micro = targets[targets[:, 1] == 0.0]
        candidate_logit = abs(float(getattr(
            self, 'geometry_oracle_candidate_logit', 12.0)))
        candidate_logits = raw.new_full(
            (batch, anchors_n, height, width), -candidate_logit)
        center_override = raw.new_full(
            (batch, anchors_n, height, width, 2), float('nan'))

        image_wh = raw.new_tensor([
            width * float(self.stride[level]),
            height * float(self.stride[level]),
        ]).float()
        all_anchor_pixels = (
            self.anchors.float() *
            self.stride.to(raw.device).float()[:, None, None]
        ).reshape(-1, 2)
        for target in macro:
            batch_index = int(target[0].item())
            target_wh = (target[4:6] * image_wh).clamp_min(1e-4)
            ratio = torch.maximum(
                target_wh[None, :] / all_anchor_pixels,
                all_anchor_pixels / target_wh[None, :])
            best_flat = int(ratio.amax(dim=1).argmin().item())
            best_level = best_flat // self.na
            best_anchor = best_flat % self.na
            if best_level != level:
                continue
            center_x = float(target[2].item() * width)
            center_y = float(target[3].item() * height)
            grid_x = min(max(int(center_x), 0), width - 1)
            grid_y = min(max(int(center_y), 0), height - 1)
            candidate_logits[
                batch_index, best_anchor, grid_y, grid_x
            ] = candidate_logit
            center_override[
                batch_index, best_anchor, grid_y, grid_x
            ] = raw.new_tensor([center_x, center_y])

        support = raw.new_zeros((batch, 1, height, width))
        if micro.numel():
            yy, xx = torch.meshgrid(
                torch.arange(height, device=raw.device, dtype=torch.float32),
                torch.arange(width, device=raw.device, dtype=torch.float32),
                indexing='ij')
            sigma = max(float(getattr(
                self, 'geometry_oracle_point_sigma', 0.75)), 0.1)
            denominator = 2.0 * sigma * sigma
            for target in micro:
                batch_index = int(target[0].item())

                point_x = target[2] * width - 0.5
                point_y = target[3] * height - 0.5
                gaussian = torch.exp(
                    -((xx - point_x).square() +
                      (yy - point_y).square()) / denominator)
                support[batch_index, 0] = torch.maximum(
                    support[batch_index, 0], gaussian.to(support.dtype))

        return {
            'support_prob': support.clamp(0.0, 1.0),
            'candidate_logits': candidate_logits,
            'center_override': center_override,
        }

    def _geometric_refine_level(self, raw, support_prob, level,
                                candidate_logits=None, center_override=None):
        """Expand a decoded Fourier contour with point support and re-project it.

        ``raw`` uses the native training parameterization. The method decodes
        its shape coefficients into feature-map units, builds a radial soft
        union with the point-support map, extracts the outer envelope, applies
        a fixed DFT, and returns coefficients encoded in the same raw space.
        The center channels are intentionally unchanged.
        """
        cm = getattr(self, 'cm', None)
        if (not bool(getattr(self, 'support_residual', False)) or
                not torch.is_tensor(cm) or cm.numel() != self.ft_coef or
                support_prob is None or support_prob.ndim != 4 or
                not torch.is_tensor(candidate_logits)):
            return raw, None

        batch, anchors_n, height, width, length = raw.shape
        if support_prob.shape[0] != batch:


            if support_prob.shape[0] >= batch:
                support_prob = support_prob[-batch:]
            else:
                return raw, None
        if support_prob.shape[-2:] != (height, width):
            support_prob = F.interpolate(
                support_prob, size=(height, width), mode='bilinear', align_corners=False)
        if candidate_logits.ndim == 5 and candidate_logits.shape[-1] == 1:
            candidate_logits = candidate_logits[..., 0]
        if candidate_logits.shape[0] != batch:
            if candidate_logits.shape[0] >= batch:
                candidate_logits = candidate_logits[-batch:]
            else:
                return raw, None
        if (candidate_logits.ndim != 4 or
                candidate_logits.shape[:2] != (batch, anchors_n)):
            return raw, None
        if candidate_logits.shape[-2:] != (height, width):
            candidate_logits = F.interpolate(
                candidate_logits.float(), size=(height, width),
                mode='bilinear', align_corners=False)
        if torch.is_tensor(center_override):
            if center_override.shape[0] != batch:
                if center_override.shape[0] >= batch:
                    center_override = center_override[-batch:]
                else:
                    center_override = None
            if (torch.is_tensor(center_override) and
                    center_override.shape !=
                    (batch, anchors_n, height, width, 2)):
                center_override = None

        directions_n, radial_n, support_threshold, temperature, outward_bias,\
            radius_scale, presence_gain, chunk_size = self._geometric_settings()
        candidate_threshold, gate_threshold, gate_temperature, candidate_topk,\
            residual_gain, debug_topk = self._candidate_geometric_settings()
        work = raw.float()
        support = support_prob.float().clamp(0.0, 1.0)
        oracle_geometry_active = bool(
            getattr(self, 'geometry_oracle', False) and
            center_override is not None)
        mass_kernel = max(int(getattr(
            self, 'geometry_support_mass_kernel', 1)), 1)
        mass_threshold = max(float(getattr(
            self, 'geometry_support_mass_threshold', 0.0)), 0.0)
        mass_temperature = max(float(getattr(
            self, 'geometry_support_mass_temperature', 0.15)), 1e-3)
        support_active_map = (
            (support - support_threshold) /
            max(1.0 - support_threshold, 1e-3)
        ).clamp(0.0, 1.0)
        if (not oracle_geometry_active and
                mass_kernel > 1 and mass_threshold > 0.0):
            local_support_mass = F.avg_pool2d(
                support_active_map, mass_kernel, stride=1,
                padding=mass_kernel // 2) * float(mass_kernel ** 2)
            support_reliability = torch.sigmoid(
                (local_support_mass - mass_threshold) /
                mass_temperature)
            filtered_support = support * support_reliability
        else:
            local_support_mass = support_active_map
            support_reliability = torch.ones_like(support)
            filtered_support = support
        point_mode, peak_threshold, peak_nms_kernel, peak_topk,\
            candidate_roi_scale, candidate_roi_min_pixels,\
            candidate_roi_max_pixels, peak_padding_pixels,\
            discrete_center_blend, peak_min_count, peak_outlier_mad_gain =\
            self._discrete_point_settings()


        discrete_geometry_active = (
            point_mode == 'discrete' and not oracle_geometry_active)
        if discrete_geometry_active:
            discrete_peak_map = self._discrete_peak_support(
                support, support_reliability,
                peak_threshold, peak_nms_kernel)
            peak_cache = getattr(
                self, '_discrete_reference_peak_cache', None)
            if (not isinstance(peak_cache, dict) or
                    not torch.is_tensor(peak_cache.get('map'))):
                peak_cache = {
                    'map': discrete_peak_map.detach(),
                    'resolution': (int(height), int(width)),
                }
                self._discrete_reference_peak_cache = peak_cache
            reference_peak_map = peak_cache['map']
            reference_peak_resolution = tuple(
                peak_cache['resolution'])
            if reference_peak_map.shape[0] != batch:
                reference_peak_map = (
                    reference_peak_map[-batch:]
                    if reference_peak_map.shape[0] >= batch else
                    discrete_peak_map)
                reference_peak_resolution = (
                    int(reference_peak_map.shape[-2]),
                    int(reference_peak_map.shape[-1]))
        else:
            discrete_peak_map = torch.zeros_like(support)
            reference_peak_map = discrete_peak_map
            reference_peak_resolution = (int(height), int(width))
        candidate_probability = candidate_logits.float().sigmoid().detach()
        device = work.device

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32), indexing='ij')
        cells = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            anchors_n, -1, -1, -1).reshape(-1, 2)
        anchor_ids = torch.arange(
            anchors_n, device=device).view(anchors_n, 1, 1)
        anchor_ids = anchor_ids.expand(
            -1, height, width).reshape(-1)
        flat = work.reshape(batch, -1, length)
        refined = flat.clone()
        center_override_flat = (
            center_override.reshape(batch, -1, 2)
            if torch.is_tensor(center_override) else None)
        candidate_flat = candidate_probability.reshape(batch, -1)

        angles = torch.arange(directions_n, device=device, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / directions_n)
        canonical = torch.stack((angles.cos(), angles.sin()), dim=-1)
        harmonics = torch.arange(1, self.ft_coef + 1, device=device, dtype=torch.float32)
        harmonic_angles = harmonics[:, None] * angles[None, :]
        cos_basis, sin_basis = harmonic_angles.cos(), harmonic_angles.sin()
        radial_unit = torch.linspace(0.0, 1.0, radial_n, device=device, dtype=torch.float32)
        outward_prior = torch.exp(
            outward_bias * radial_unit).view(1, 1, radial_n)

        harmonic_scale = (float(self.ec) + cm.to(device=device, dtype=torch.float32)).view(1, self.ft_coef, 1)
        anchor_wh = self.anchors[level].to(device=device, dtype=torch.float32)
        anchor_scale = torch.stack(
            (anchor_wh[:, 0], anchor_wh[:, 0], anchor_wh[:, 1], anchor_wh[:, 1]), dim=-1)
        max_map_radius = 0.75 * math.sqrt(float(height * height + width * width))
        expansion_values = []
        expansion_mean_map = work.new_zeros((batch, flat.shape[1]))
        expansion_max_map = work.new_zeros((batch, flat.shape[1]))
        assigned_peak_map = work.new_zeros((batch, 1, height, width))
        assigned_peak_count_map = work.new_zeros(
            (batch, flat.shape[1]))
        discrete_center_shift_map = work.new_zeros(
            (batch, flat.shape[1]))
        debug_records = []


        for batch_idx in range(batch):
            peak_values = work.new_zeros((0,))
            peak_flat_ids = torch.zeros(
                (0,), device=device, dtype=torch.long)
            peak_coordinates = work.new_zeros((0, 2))
            if discrete_geometry_active:
                peak_flat = reference_peak_map[
                    batch_idx, 0].reshape(-1)
                point_topk = min(peak_topk, peak_flat.numel())
                if point_topk > 0:
                    peak_values, peak_flat_ids = torch.topk(
                        peak_flat, k=point_topk, largest=True)
                    peak_valid = peak_values >= peak_threshold
                    peak_values = peak_values[peak_valid]
                    peak_flat_ids = peak_flat_ids[peak_valid]
                    reference_h, reference_w = (
                        reference_peak_resolution)
                    peak_y = torch.div(
                        peak_flat_ids, reference_w,
                        rounding_mode='floor')
                    peak_x = peak_flat_ids.remainder(reference_w)
                    peak_coordinates = torch.stack((
                        (peak_x.to(dtype=torch.float32) + 0.5) *
                        float(width) / float(reference_w),
                        (peak_y.to(dtype=torch.float32) + 0.5) *
                        float(height) / float(reference_h),
                    ), dim=-1)
            selected_n = min(candidate_topk, flat.shape[1])
            scores, selected_ids = torch.topk(
                candidate_flat[batch_idx], k=selected_n, largest=True)
            keep = scores >= candidate_threshold
            scores, selected_ids = scores[keep], selected_ids[keep]
            if selected_ids.numel() == 0:
                continue

            for start in range(0, selected_ids.numel(), chunk_size):
                end = min(start + chunk_size, selected_ids.numel())
                flat_ids = selected_ids[start:end]
                chunk_scores = scores[start:end]
                raw_chunk = flat[batch_idx, flat_ids]
                ids = anchor_ids[flat_ids]
                coefficient_scale = (
                    anchor_scale[ids, None, :] * harmonic_scale)
                base_coefficients = (
                    raw_chunk[..., 2:].view(-1, self.ft_coef, 4) *
                    coefficient_scale)

                offset_x = torch.einsum(
                    'nka,kd->nd', base_coefficients[..., 0:1], cos_basis)
                offset_x = offset_x + torch.einsum(
                    'nka,kd->nd', base_coefficients[..., 1:2], sin_basis)
                offset_y = torch.einsum(
                    'nka,kd->nd', base_coefficients[..., 2:3], cos_basis)
                offset_y = offset_y + torch.einsum(
                    'nka,kd->nd', base_coefficients[..., 3:4], sin_basis)
                base_offsets = torch.stack((offset_x, offset_y), dim=-1)
                point_radius = torch.linalg.vector_norm(
                    base_offsets, dim=-1).clamp_min(1e-6)
                point_direction = base_offsets / point_radius.unsqueeze(-1)


                angular = torch.einsum(
                    'ndq,kq->nkd', point_direction, canonical)
                projection_to_ray = torch.einsum(
                    'ndq,kq->nkd', base_offsets, canonical).clamp_min(0.0)
                angular_weight = (
                    20.0 * (angular - 1.0) +
                    4.0 * point_radius[:, None, :] /
                    point_radius.amax(dim=-1, keepdim=True)[:, :, None]
                    .clamp_min(1e-6)
                ).softmax(dim=-1)
                base_radius = (
                    angular_weight * projection_to_ray).sum(dim=-1)

                predicted_centers = (
                    cells[flat_ids] + 0.5 + raw_chunk[..., :2])
                point_centers = predicted_centers.new_full(
                    predicted_centers.shape, float('nan'))
                center_blend = 0.0
                if center_override_flat is not None:
                    oracle_centers = center_override_flat[
                        batch_idx, flat_ids]
                    point_centers = oracle_centers
                    has_oracle_center = torch.isfinite(
                        oracle_centers).all(dim=-1, keepdim=True)
                    center_blend = (
                        1.0 if bool(getattr(
                            self, 'geometry_oracle', False)) else
                        float(getattr(
                            self, 'geometry_center_blend', 1.0)))
                    safe_override_centers = torch.where(
                        has_oracle_center, oracle_centers,
                        predicted_centers)
                    corrected_centers = (
                        predicted_centers +
                        center_blend *
                        (safe_override_centers - predicted_centers))
                    centers = torch.where(
                        has_oracle_center, corrected_centers,
                        predicted_centers)
                    refined[batch_idx, flat_ids, :2] = (
                        centers - (cells[flat_ids] + 0.5))
                else:
                    centers = predicted_centers
                anchor_radius = 0.75 * torch.linalg.vector_norm(
                    anchor_wh[ids], dim=-1)
                sample_radius = torch.maximum(
                    base_radius.amax(dim=-1) * radius_scale,
                    anchor_radius * radius_scale).clamp(
                        min=1.0, max=max_map_radius)
                candidate_count = flat_ids.numel()
                peak_count = work.new_zeros((candidate_count,))
                point_envelope = base_radius.clone()
                point_extent_centers = centers.new_full(
                    centers.shape, float('nan'))
                point_assignment_radius = sample_radius.clone()
                point_center_shift = work.new_zeros((candidate_count,))
                assigned_peak_xy = centers.new_full(
                    (candidate_count, peak_coordinates.shape[0], 2),
                    float('nan'))
                assigned_peak_scores = work.new_zeros(
                    (candidate_count, peak_coordinates.shape[0]))
                valid_point_set = torch.zeros(
                    (candidate_count,), device=device, dtype=torch.bool)

                if (discrete_geometry_active and
                        peak_coordinates.numel() > 0):
                    stride_value = max(
                        float(self.stride[level]), 1e-6)
                    roi_min = candidate_roi_min_pixels / stride_value
                    roi_max = candidate_roi_max_pixels / stride_value
                    point_assignment_radius = (
                        base_radius.amax(dim=-1) *
                        candidate_roi_scale
                    ).clamp(min=roi_min, max=max(roi_max, roi_min))

                    initial_delta = (
                        peak_coordinates[None, :, :] -
                        centers[:, None, :])
                    initial_distance = torch.linalg.vector_norm(
                        initial_delta, dim=-1)
                    normalized_distance = (
                        initial_distance /
                        point_assignment_radius[:, None].clamp_min(1e-6))
                    eligible = normalized_distance <= 1.0
                    masked_distance = normalized_distance.masked_fill(
                        ~eligible, float('inf'))
                    nearest_candidate = masked_distance.argmin(dim=0)
                    has_candidate = eligible.any(dim=0)
                    assignment = F.one_hot(
                        nearest_candidate,
                        num_classes=candidate_count
                    ).transpose(0, 1).to(dtype=torch.bool)
                    assignment = assignment & has_candidate[None, :]


                    assigned_count = assignment.sum(dim=-1)
                    if peak_outlier_mad_gain > 0.0:
                        distance_nan = initial_distance.masked_fill(
                            ~assignment, float('nan'))
                        radial_median = torch.nanmedian(
                            distance_nan, dim=-1).values
                        radial_mad = torch.nanmedian(
                            (initial_distance -
                             radial_median[:, None]).abs().masked_fill(
                                ~assignment, float('nan')),
                            dim=-1).values
                        radial_median = torch.nan_to_num(
                            radial_median, nan=0.0)
                        radial_mad = torch.nan_to_num(
                            radial_mad, nan=0.0)
                        robust_limit = torch.maximum(
                            radial_median +
                            peak_outlier_mad_gain * radial_mad,
                            0.85 * point_assignment_radius)
                        robust_limit = torch.minimum(
                            robust_limit, point_assignment_radius)
                        apply_robust = assigned_count >= 4
                        assignment = assignment & (
                            (~apply_robust[:, None]) |
                            (initial_distance <= robust_limit[:, None]))

                    peak_count = assignment.sum(dim=-1).to(
                        dtype=work.dtype)
                    valid_point_set = peak_count >= float(peak_min_count)
                    coordinate_bank = peak_coordinates[None, :, :].expand(
                        candidate_count, -1, -1)
                    coordinate_min = coordinate_bank.masked_fill(
                        ~assignment[..., None], float('inf')).amin(dim=1)
                    coordinate_max = coordinate_bank.masked_fill(
                        ~assignment[..., None], float('-inf')).amax(dim=1)
                    point_extent_centers = 0.5 * (
                        coordinate_min + coordinate_max)
                    point_extent_centers = torch.where(
                        valid_point_set[:, None],
                        point_extent_centers,
                        centers.new_full(centers.shape, float('nan')))


                    safe_extent_center = torch.where(
                        valid_point_set[:, None],
                        point_extent_centers, centers)
                    discrete_centers = (
                        centers +
                        discrete_center_blend *
                        (safe_extent_center - centers))
                    point_center_shift = torch.linalg.vector_norm(
                        discrete_centers - centers, dim=-1)
                    centers = torch.where(
                        valid_point_set[:, None],
                        discrete_centers, centers)
                    point_centers = torch.where(
                        valid_point_set[:, None],
                        point_extent_centers, point_centers)
                    refined[batch_idx, flat_ids, :2] = (
                        centers - (cells[flat_ids] + 0.5))

                    point_delta = (
                        peak_coordinates[None, :, :] -
                        centers[:, None, :])
                    point_projection = torch.einsum(
                        'npq,kq->nkp', point_delta, canonical)
                    point_projection = point_projection.masked_fill(
                        ~assignment[:, None, :], float('-inf'))
                    point_envelope = torch.nan_to_num(
                        point_projection.amax(dim=-1),
                        nan=0.0, neginf=0.0, posinf=0.0)
                    point_envelope = (
                        point_envelope.clamp_min(0.0) +
                        peak_padding_pixels / stride_value)
                    point_envelope = torch.where(
                        valid_point_set[:, None],
                        point_envelope, base_radius)
                    support_radius = point_envelope
                    raw_expansion = F.relu(
                        point_envelope - base_radius)
                    outside_evidence = (
                        (raw_expansion > 0.0).to(dtype=work.dtype) *
                        valid_point_set[:, None].to(dtype=work.dtype)
                    ).unsqueeze(-1)

                    assigned_peak_xy = coordinate_bank.masked_fill(
                        ~assignment[..., None], float('nan'))
                    assigned_peak_scores = (
                        peak_values[None, :].expand(
                            candidate_count, -1) *
                        assignment.to(dtype=work.dtype))
                    accepted_peak = (
                        assignment &
                        valid_point_set[:, None]).any(dim=0)
                    if accepted_peak.any():
                        assigned_flat = assigned_peak_map[
                            batch_idx, 0].view(-1)
                        accepted_xy = peak_coordinates[accepted_peak]
                        accepted_x = accepted_xy[:, 0].floor().long().clamp(
                            min=0, max=width - 1)
                        accepted_y = accepted_xy[:, 1].floor().long().clamp(
                            min=0, max=height - 1)
                        assigned_ids = accepted_y * width + accepted_x
                        assigned_flat[assigned_ids] = peak_values[
                            accepted_peak]
                    assigned_peak_count_map[
                        batch_idx, flat_ids] = peak_count
                    discrete_center_shift_map[
                        batch_idx, flat_ids] = point_center_shift
                else:
                    radii = (
                        sample_radius[:, None, None] *
                        radial_unit.view(1, 1, radial_n))
                    sample_xy = (
                        centers[:, None, None, :] +
                        canonical[None, :, None, :] *
                        radii.unsqueeze(-1))
                    grid_x = (
                        (sample_xy[..., 0] - 0.5) *
                        (2.0 / max(width - 1, 1)) - 1.0)
                    grid_y = (
                        (sample_xy[..., 1] - 0.5) *
                        (2.0 / max(height - 1, 1)) - 1.0)
                    sample_grid = torch.stack(
                        (grid_x, grid_y), dim=-1).reshape(
                            1, -1, radial_n, 2)
                    sampled_support = F.grid_sample(
                        filtered_support[batch_idx:batch_idx + 1],
                        sample_grid, mode='bilinear',
                        padding_mode='zeros',
                        align_corners=True).view(
                            -1, directions_n, radial_n)
                    active_support = (
                        (sampled_support - support_threshold) /
                        max(1.0 - support_threshold, 1e-3)
                    ).clamp(0.0, 1.0)

                    base_region = torch.sigmoid(
                        (base_radius.unsqueeze(-1) - radii) /
                        temperature)
                    outside_evidence = (
                        active_support * (1.0 - base_region))
                    envelope_weight = outside_evidence * outward_prior
                    weight_sum = envelope_weight.sum(dim=-1)
                    support_radius = (
                        (envelope_weight * radii).sum(dim=-1) /
                        weight_sum.clamp_min(1e-8))
                    directional_presence = 1.0 - torch.exp(
                        -presence_gain *
                        outside_evidence.sum(dim=-1))
                    raw_expansion = directional_presence * F.relu(
                        support_radius - base_radius)
                    point_envelope = support_radius
                expansion = raw_expansion
                angular_kernel = max(int(getattr(
                    self, 'geometry_angular_smooth_kernel', 1)), 1)
                if angular_kernel > 1:
                    if not angular_kernel % 2:
                        angular_kernel += 1
                    angular_padding = angular_kernel // 2
                    angular_average = F.avg_pool1d(
                        F.pad(
                            expansion.unsqueeze(1),
                            (angular_padding, angular_padding),
                            mode='circular'),
                        angular_kernel, stride=1).squeeze(1)
                    expansion = torch.minimum(
                        expansion,
                        angular_average * float(getattr(
                            self, 'geometry_angular_spike_gain', 2.0)))
                expansion_cap_ratio = max(float(getattr(
                    self, 'geometry_expansion_cap_ratio', 0.0)), 0.0)
                if expansion_cap_ratio > 0.0:
                    expansion = torch.minimum(
                        expansion,
                        base_radius * expansion_cap_ratio)
                final_radius = base_radius + expansion
                final_offsets = (
                    canonical[None, :, :] * final_radius.unsqueeze(-1))

                projection = 2.0 / float(directions_n)
                projected = torch.stack((
                    projection * torch.einsum(
                        'nd,kd->nk', final_offsets[..., 0], cos_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', final_offsets[..., 0], sin_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', final_offsets[..., 1], cos_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', final_offsets[..., 1], sin_basis),
                ), dim=-1)
                candidate_gate = torch.sigmoid(
                    (chunk_scores - gate_threshold) / gate_temperature)
                if discrete_geometry_active:


                    candidate_gate = torch.where(
                        valid_point_set,
                        torch.ones_like(candidate_gate),
                        candidate_gate)
                outside_presence = 1.0 - torch.exp(
                    -presence_gain * outside_evidence.sum(dim=(1, 2)))
                blend = (
                    residual_gain * candidate_gate * outside_presence
                ).clamp(0.0, 1.0)
                refined_coefficients = (
                    base_coefficients +
                    blend[:, None, None] *
                    (projected - base_coefficients))
                encoded = (
                    refined_coefficients /
                    coefficient_scale.clamp_min(1e-8))
                refined[batch_idx, flat_ids, 2:] = encoded.reshape(
                    -1, length - 2)
                expansion_mean_map[batch_idx, flat_ids] = (
                    expansion.detach().mean(dim=-1))
                expansion_max_map[batch_idx, flat_ids] = (
                    expansion.detach().amax(dim=-1))
                expansion_values.append(expansion.detach().mean())

                remaining_debug = debug_topk - sum(
                    record['scores'].numel()
                    for record in debug_records
                    if record['batch'] == batch_idx)
                if remaining_debug > 0:
                    take = min(remaining_debug, flat_ids.numel())
                    debug_records.append({
                        'batch': batch_idx,
                        'flat_ids': flat_ids[:take].detach(),
                        'scores': chunk_scores[:take].detach(),
                        'centers': centers[:take].detach(),
                        'predicted_centers': (
                            predicted_centers[:take].detach()),
                        'point_centers': point_centers[:take].detach(),
                        'point_extent_centers': (
                            point_extent_centers[:take].detach()),
                        'center_blend': centers.new_full(
                            (take,), float(center_blend)).detach(),
                        'discrete_center_blend': centers.new_full(
                            (take,), float(
                                discrete_center_blend
                                if discrete_geometry_active else 0.0)
                        ).detach(),
                        'discrete_center_shift': (
                            point_center_shift[:take].detach()),
                        'point_assignment_radius': (
                            point_assignment_radius[:take].detach()),
                        'assigned_peak_count': (
                            peak_count[:take].detach()),
                        'assigned_peak_xy': (
                            assigned_peak_xy[:take].detach()),
                        'assigned_peak_scores': (
                            assigned_peak_scores[:take].detach()),
                        'base_radius': base_radius[:take].detach(),
                        'support_radius': support_radius[:take].detach(),
                        'point_envelope': point_envelope[:take].detach(),
                        'raw_expansion': raw_expansion[:take].detach(),
                        'expansion': expansion[:take].detach(),
                        'final_radius': final_radius[:take].detach(),
                        'base_coefficients': base_coefficients[:take].detach(),
                        'refined_coefficients': (
                            refined_coefficients[:take].detach()),
                        'blend': blend[:take].detach(),
                    })

        geometry_store = GLOBAL_CANDIDATE_GEOMETRY_DICT.get(device)
        if not isinstance(geometry_store, dict):
            geometry_store = {}
        resolution = (int(height), int(width))
        geometry_entry = geometry_store.get(resolution, {})
        geometry_entry.update({
            'candidate_prob': candidate_probability.detach(),
            'support_prob': support.detach(),
            'filtered_support_prob': filtered_support.detach(),
            'support_reliability': support_reliability.detach(),
            'support_local_mass': local_support_mass.detach(),
            'discrete_peak_map': discrete_peak_map.detach(),
            'discrete_reference_peak_map': (
                reference_peak_map.detach()),
            'assigned_peak_map': assigned_peak_map.detach(),
            'assigned_peak_count': assigned_peak_count_map.view(
                batch, anchors_n, height, width).detach(),
            'discrete_center_shift': discrete_center_shift_map.view(
                batch, anchors_n, height, width).detach(),
            'expansion_mean': expansion_mean_map.view(
                batch, anchors_n, height, width).detach(),
            'expansion_max': expansion_max_map.view(
                batch, anchors_n, height, width).detach(),
            'directions': angles.detach(),
            'topk_records': debug_records,
            'oracle_mode': oracle_geometry_active,
            'point_mode': point_mode,
        })
        geometry_store[resolution] = geometry_entry
        GLOBAL_CANDIDATE_GEOMETRY_DICT[device] = geometry_store

        stats = {
            'mean_expansion': (
                torch.stack(expansion_values).mean()
                if expansion_values else work.new_zeros(())),
            'max_expansion': expansion_max_map.amax().detach(),
            'selected_candidates': torch.tensor(
                sum(record['scores'].numel() for record in debug_records),
                device=device),
            'assigned_peaks': assigned_peak_count_map.sum().detach(),
        }
        return refined.reshape_as(work).to(dtype=raw.dtype), stats

    def _candidate_conditioned_point_spectral_level(
            self, raw, feature, level):
        """Map dense point evidence to a gated residual in Fourier space.

        The native Fourier regression in ``raw`` is always retained.  For a
        small set of proposal-routed candidates, the live point probability is
        softly assigned to the candidate ROI and summarized as a directional
        radial support.  A fixed DFT converts that support into the exact
        coefficient parameterization used by this head.  Candidate confidence,
        point-set reliability and a learned per-order gate jointly determine
        the bounded residual:

            C_final[k] = C_native[k] + g[k] * (C_point[k] - C_native[k]).

        No peak extraction, NMS, polygon construction or label substitution is
        used here.  Point/Fourier losses therefore backpropagate through the
        dense point probability, the proposal score and the spectral gates.
        """
        if (not bool(getattr(self, 'point_spectral_enabled', False)) or
                not torch.is_tensor(getattr(self, 'cm', None)) or
                len(getattr(self, 'point_spectral_gate_heads', ())) != self.nl):
            return raw, None

        point_logits = GLOBAL_POINT_DICT.get(raw.device)
        proposal_store = GLOBAL_FOURIER_PROPOSAL_DICT.get(raw.device)
        proposal_entry = (
            proposal_store.get(tuple(raw.shape[2:4]))
            if isinstance(proposal_store, dict) else None)
        candidate_logits = (
            proposal_entry.get('live_logits')
            if isinstance(proposal_entry, dict) else None)
        if (not torch.is_tensor(point_logits) or
                not torch.is_tensor(candidate_logits)):
            return raw, None

        batch, anchors_n, height, width, length = raw.shape
        if point_logits.shape[0] != batch:
            point_logits = (
                point_logits[-batch:]
                if point_logits.shape[0] >= batch else None)
        if candidate_logits.shape[0] != batch:
            candidate_logits = (
                candidate_logits[-batch:]
                if candidate_logits.shape[0] >= batch else None)
        if point_logits is None or candidate_logits is None:
            return raw, None
        if point_logits.shape[-2:] != (height, width):
            point_logits = F.interpolate(
                point_logits, size=(height, width), mode='bilinear',
                align_corners=False)
        if candidate_logits.ndim == 5 and candidate_logits.shape[-1] == 1:
            candidate_logits = candidate_logits[..., 0]
        if (candidate_logits.ndim != 4 or
                candidate_logits.shape[1:] !=
                (anchors_n, height, width)):
            return raw, None

        work = raw.float()
        point_probability = point_logits.float().sigmoid()
        candidate_probability = candidate_logits.float().sigmoid()
        point_grad_scale = min(max(float(getattr(
            self, 'point_spectral_point_grad_scale', 1.0)), 0.0), 1.0)


        bridge_point_probability = (
            point_probability.detach() + point_grad_scale *
            (point_probability - point_probability.detach()))
        point_threshold = float(getattr(
            self, 'point_spectral_point_threshold', 0.20))
        point_temperature = max(float(getattr(
            self, 'point_spectral_point_temperature', 0.08)), 1e-3)


        active_point = bridge_point_probability * torch.sigmoid(
            (bridge_point_probability - point_threshold) / point_temperature)

        gate_logits = self.point_spectral_gate_heads[level](feature).float()
        gate_logits = gate_logits.view(
            batch, anchors_n, self.ft_coef, height, width
        ).permute(0, 1, 3, 4, 2).reshape(
            batch, -1, self.ft_coef)

        candidate_flat = candidate_probability.reshape(batch, -1)
        routed_n = min(
            max(int(getattr(
                self, 'point_spectral_candidate_topk', 12)), 1),
            candidate_flat.shape[1])
        routed_scores, routed_ids = torch.topk(
            candidate_flat, k=routed_n, dim=1, largest=True)
        candidate_threshold = float(getattr(
            self, 'point_spectral_candidate_threshold', 0.05))
        candidate_temperature = max(float(getattr(
            self, 'point_spectral_candidate_temperature', 0.08)), 1e-3)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=raw.device, dtype=torch.float32),
            torch.arange(width, device=raw.device, dtype=torch.float32),
            indexing='ij')
        spatial_xy = torch.stack(
            (xx + 0.5, yy + 0.5), dim=-1).reshape(-1, 2)
        cells = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            anchors_n, -1, -1, -1).reshape(-1, 2)
        anchor_ids = torch.arange(
            anchors_n, device=raw.device).view(anchors_n, 1, 1)
        anchor_ids = anchor_ids.expand(
            -1, height, width).reshape(-1)

        directions_n = max(int(getattr(
            self, 'point_spectral_directions', 64)),
            2 * self.ft_coef + 1)
        angles = torch.arange(
            directions_n, device=raw.device, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / float(directions_n))
        canonical = torch.stack((angles.cos(), angles.sin()), dim=-1)
        harmonics = torch.arange(
            1, self.ft_coef + 1,
            device=raw.device, dtype=torch.float32)
        harmonic_angles = harmonics[:, None] * angles[None, :]
        cos_basis = harmonic_angles.cos()
        sin_basis = harmonic_angles.sin()
        harmonic_scale = (
            float(self.ec) + self.cm.to(
                device=raw.device, dtype=torch.float32)
        ).view(1, self.ft_coef, 1)
        anchor_wh = self.anchors[level].to(
            device=raw.device, dtype=torch.float32)
        anchor_scale = torch.stack((
            anchor_wh[:, 0], anchor_wh[:, 0],
            anchor_wh[:, 1], anchor_wh[:, 1]), dim=-1)
        order_prior = torch.exp(
            -float(getattr(self, 'point_spectral_order_decay', 0.04)) *
            torch.arange(
                self.ft_coef, device=raw.device, dtype=torch.float32))

        flat = work.reshape(batch, -1, length)
        refined = flat.clone()
        assignment_map = work.new_zeros((batch, 1, height, width))
        reliability_map = work.new_zeros(
            (batch, anchors_n, height, width))
        gate_map = work.new_zeros((batch, anchors_n, height, width))
        expansion_map = work.new_zeros((batch, anchors_n, height, width))
        debug_records = []
        reliability_values = []
        gate_values = []

        roi_scale = max(float(getattr(
            self, 'point_spectral_roi_scale', 2.25)), 1.0)
        stride_value = max(float(self.stride[level]), 1e-6)
        roi_min = float(getattr(
            self, 'point_spectral_roi_min_pixels', 32.0)) / stride_value
        roi_max = float(getattr(
            self, 'point_spectral_roi_max_pixels', 96.0)) / stride_value
        assignment_temperature = max(float(getattr(
            self, 'point_spectral_assignment_temperature', 0.12)), 1e-3)
        direction_temperature = max(float(getattr(
            self, 'point_spectral_direction_temperature', 0.10)), 1e-3)
        min_mass = max(float(getattr(
            self, 'point_spectral_min_mass', 0.75)), 0.0)
        mass_temperature = max(float(getattr(
            self, 'point_spectral_mass_temperature', 0.25)), 1e-3)
        max_gain = min(max(float(getattr(
            self, 'point_spectral_max_gain', 0.75)), 0.0), 1.0)
        radial_softmax_beta = max(float(getattr(
            self, 'point_spectral_radial_softmax_beta', 0.0)), 0.0)
        gate_floor = torch.as_tensor(
            getattr(
                self, 'point_spectral_gate_floor',
                torch.zeros(self.ft_coef, device=raw.device)),
            device=raw.device, dtype=torch.float32).flatten()
        if gate_floor.numel() == 1:
            gate_floor = gate_floor.repeat(self.ft_coef)
        elif gate_floor.numel() < self.ft_coef:
            gate_floor = F.pad(
                gate_floor, (0, self.ft_coef - gate_floor.numel()))
        gate_floor = gate_floor[:self.ft_coef].clamp(0.0, 1.0)
        padding = max(float(getattr(
            self, 'point_spectral_padding_pixels', 2.0)), 0.0) / stride_value
        cap_ratio = max(float(getattr(
            self, 'point_spectral_expansion_cap_ratio', 1.50)), 0.0)
        min_cap_roi_ratio = max(float(getattr(
            self, 'point_spectral_min_cap_roi_ratio', 0.0)), 0.0)
        fusion_mode = str(getattr(
            self, 'point_spectral_fusion_mode',
            'coefficient_blend')).strip().lower()
        debug_topk = max(int(getattr(
            self, 'point_spectral_debug_topk', 4)), 1)

        for batch_idx in range(batch):
            ids = routed_ids[batch_idx]
            scores = routed_scores[batch_idx]


            raw_selected = flat[batch_idx, ids]
            selected_anchor = anchor_ids[ids]
            coefficient_scale = (
                anchor_scale[selected_anchor, None, :] * harmonic_scale)
            base_coefficients = (
                raw_selected[..., 2:].view(-1, self.ft_coef, 4) *
                coefficient_scale)
            offset_x = (
                torch.einsum('nk,kd->nd', base_coefficients[..., 0], cos_basis) +
                torch.einsum('nk,kd->nd', base_coefficients[..., 1], sin_basis))
            offset_y = (
                torch.einsum('nk,kd->nd', base_coefficients[..., 2], cos_basis) +
                torch.einsum('nk,kd->nd', base_coefficients[..., 3], sin_basis))
            base_offsets = torch.stack((offset_x, offset_y), dim=-1)
            contour_norm = torch.linalg.vector_norm(
                base_offsets, dim=-1).clamp_min(1e-6)
            contour_unit = base_offsets / contour_norm.unsqueeze(-1)
            angular_similarity = torch.einsum(
                'ndq,kq->nkd', contour_unit, canonical)
            contour_projection = torch.einsum(
                'ndq,kq->nkd', base_offsets, canonical).clamp_min(0.0)
            contour_weight = (
                20.0 * (angular_similarity - 1.0) +
                4.0 * contour_norm[:, None, :] /
                contour_norm.amax(dim=-1, keepdim=True)[:, :, None]
                .clamp_min(1e-6)
            ).softmax(dim=-1)
            base_radius = (contour_weight * contour_projection).sum(dim=-1)

            centers = cells[ids] + 0.5 + raw_selected[..., :2]
            anchor_radius = 0.5 * torch.linalg.vector_norm(
                anchor_wh[selected_anchor], dim=-1)
            roi_radius = torch.maximum(
                base_radius.amax(dim=-1) * roi_scale,
                anchor_radius).clamp(min=roi_min, max=max(roi_max, roi_min))


            spatial_delta = spatial_xy[None, :, :] - centers[:, None, :]
            spatial_distance = torch.linalg.vector_norm(
                spatial_delta, dim=-1)
            spatial_normalized = (
                spatial_distance / roi_radius[:, None].clamp_min(1e-6))
            roi_assignment = torch.sigmoid(
                (1.0 - spatial_normalized) / assignment_temperature)
            spatial_point_weight = (
                active_point[batch_idx, 0].reshape(1, -1) * roi_assignment)
            point_mass = spatial_point_weight.sum(dim=-1)
            reliability = torch.sigmoid(
                (point_mass - min_mass) / mass_temperature)

            radial_n = max(int(getattr(
                self, 'point_spectral_radial_samples', 24)), 8)
            radial_unit = torch.linspace(
                0.0, 1.0, radial_n,
                device=raw.device, dtype=torch.float32)
            radii = (
                roi_radius[:, None, None] *
                radial_unit[None, None, :]
            ).expand(-1, directions_n, -1)
            sample_xy = (
                centers[:, None, None, :] +
                canonical[None, :, None, :] * radii[..., None])
            sample_grid_x = (
                (sample_xy[..., 0] - 0.5) *
                (2.0 / max(width - 1, 1)) - 1.0)
            sample_grid_y = (
                (sample_xy[..., 1] - 0.5) *
                (2.0 / max(height - 1, 1)) - 1.0)
            sample_grid = torch.stack(
                (sample_grid_x, sample_grid_y), dim=-1
            ).reshape(1, -1, radial_n, 2)
            sampled_point = F.grid_sample(
                active_point[batch_idx:batch_idx + 1],
                sample_grid, mode='bilinear', padding_mode='zeros',
                align_corners=True).view(-1, directions_n, radial_n)
            direction_mix = torch.softmax(
                (canonical @ canonical.transpose(0, 1) - 1.0) /
                direction_temperature, dim=-1)
            sampled_point = torch.einsum(
                'de,ner->ndr', direction_mix, sampled_point)
            radial_assignment = torch.sigmoid(
                (1.0 - radial_unit) / assignment_temperature)
            directional_point_weight = (
                sampled_point * radial_assignment[None, None, :])
            if radial_softmax_beta > 0.0:


                radial_logits = (
                    radial_softmax_beta * radial_unit[None, None, :] +
                    directional_point_weight.clamp_min(1e-8).log())
                radial_distribution = torch.softmax(radial_logits, dim=-1)
                support_radius = (
                    radial_distribution * radii).sum(dim=-1)
            else:

                radial_weight = (
                    directional_point_weight *
                    torch.exp(4.0 * radial_unit)[None, None, :])
                radial_weight_sum = radial_weight.sum(dim=-1)
                support_radius = (
                    radial_weight * radii
                ).sum(dim=-1) / radial_weight_sum.clamp_min(1e-8)
            direction_mass = directional_point_weight.sum(dim=-1)
            direction_presence = 1.0 - torch.exp(-4.0 * direction_mass)
            raw_expansion = (
                direction_presence *
                F.relu(support_radius + padding - base_radius))


            smooth_expansion = F.avg_pool1d(
                F.pad(raw_expansion.unsqueeze(1), (1, 1), mode='circular'),
                kernel_size=3, stride=1).squeeze(1)
            expansion = torch.minimum(
                raw_expansion, 2.0 * smooth_expansion)
            if cap_ratio > 0.0:


                cap_reference = torch.maximum(
                    base_radius,
                    roi_radius[:, None] * min_cap_roi_ratio)
                expansion = torch.minimum(
                    expansion, cap_reference * cap_ratio)
            final_radius = base_radius + expansion
            final_offsets = (
                canonical[None, :, :] * final_radius.unsqueeze(-1))
            projection = 2.0 / float(directions_n)
            point_coefficients = torch.stack((
                projection * torch.einsum(
                    'nd,kd->nk', final_offsets[..., 0], cos_basis),
                projection * torch.einsum(
                    'nd,kd->nk', final_offsets[..., 0], sin_basis),
                projection * torch.einsum(
                    'nd,kd->nk', final_offsets[..., 1], cos_basis),
                projection * torch.einsum(
                    'nd,kd->nk', final_offsets[..., 1], sin_basis),
            ), dim=-1)

            learned_gate = torch.sigmoid(gate_logits[batch_idx, ids])
            candidate_gate = torch.sigmoid(
                (scores - candidate_threshold) / candidate_temperature)
            if bool((gate_floor > 0.0).any()):


                effective_gate = (
                    gate_floor[None, :] +
                    (1.0 - gate_floor[None, :]) * learned_gate)
                harmonic_gate = (
                    max_gain * candidate_gate[:, None] *
                    reliability[:, None] * effective_gate).clamp(0.0, 1.0)
            else:

                harmonic_gate = (
                    max_gain * candidate_gate[:, None] *
                    reliability[:, None] * learned_gate *
                    order_prior[None, :]).clamp(0.0, 1.0)
            if fusion_mode == 'phase_aligned_radial_union':


                valid_base_direction = contour_norm > 1e-4
                base_direction = torch.where(
                    valid_base_direction[..., None], contour_unit,
                    canonical[None, :, :])
                aligned_similarity = torch.einsum(
                    'ndq,kq->ndk', base_direction, canonical)
                aligned_direction_weight = torch.softmax(
                    (aligned_similarity - 1.0) / direction_temperature,
                    dim=-1)
                aligned_support = torch.einsum(
                    'ndk,nk->nd', aligned_direction_weight, support_radius)
                aligned_presence = torch.einsum(
                    'ndk,nk->nd', aligned_direction_weight,
                    direction_presence)
                aligned_raw_expansion = (
                    aligned_presence * F.relu(
                        aligned_support + padding - contour_norm))
                aligned_smooth_expansion = F.avg_pool1d(
                    F.pad(
                        aligned_raw_expansion.unsqueeze(1),
                        (1, 1), mode='circular'),
                    kernel_size=3, stride=1).squeeze(1)
                aligned_expansion = torch.minimum(
                    aligned_raw_expansion,
                    2.0 * aligned_smooth_expansion)
                if cap_ratio > 0.0:
                    aligned_cap_reference = torch.maximum(
                        contour_norm,
                        roi_radius[:, None] * min_cap_roi_ratio)
                    aligned_expansion = torch.minimum(
                        aligned_expansion,
                        aligned_cap_reference * cap_ratio)


                low_order_count = min(2, self.ft_coef)
                union_gate = harmonic_gate[:, :low_order_count].mean(dim=-1)
                applied_expansion = (
                    union_gate[:, None] * aligned_expansion)
                union_offsets = (
                    base_offsets +
                    base_direction * applied_expansion[..., None])


                union_center_shift = union_offsets.mean(dim=1)
                centered_union_offsets = (
                    union_offsets - union_center_shift[:, None, :])
                refined_coefficients = torch.stack((
                    projection * torch.einsum(
                        'nd,kd->nk', centered_union_offsets[..., 0], cos_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', centered_union_offsets[..., 0], sin_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', centered_union_offsets[..., 1], cos_basis),
                    projection * torch.einsum(
                        'nd,kd->nk', centered_union_offsets[..., 1], sin_basis),
                ), dim=-1)
                debug_base_radius = contour_norm
                debug_support_radius = aligned_support
                debug_raw_expansion = aligned_raw_expansion
                debug_expansion = applied_expansion
                debug_final_radius = contour_norm + applied_expansion
                debug_blend = union_gate
            else:
                refined_coefficients = (
                    base_coefficients + harmonic_gate[..., None] *
                    (point_coefficients - base_coefficients))
                debug_base_radius = base_radius
                debug_support_radius = support_radius
                debug_raw_expansion = raw_expansion
                debug_expansion = expansion
                debug_final_radius = final_radius
                debug_blend = harmonic_gate.mean(dim=-1)
                union_center_shift = None
            if union_center_shift is not None:
                refined[batch_idx, ids, :2] = (
                    raw_selected[..., :2] + union_center_shift)
            refined[batch_idx, ids, 2:] = (
                refined_coefficients /
                coefficient_scale.clamp_min(1e-8)
            ).reshape(-1, length - 2)

            assignment_map[batch_idx, 0] = torch.maximum(
                assignment_map[batch_idx, 0],
                spatial_point_weight.amax(dim=0).reshape(
                    height, width).detach())
            reliability_map[batch_idx].view(-1)[ids] = reliability.detach()
            gate_map[batch_idx].view(-1)[ids] = debug_blend.detach()
            expansion_map[batch_idx].view(-1)[ids] = (
                debug_expansion.detach().mean(dim=-1))
            reliability_values.append(reliability.detach().mean())
            gate_values.append(debug_blend.detach().mean())

            take = min(debug_topk, ids.numel())
            debug_records.append({
                'batch': batch_idx,
                'flat_ids': ids[:take].detach(),
                'scores': scores[:take].detach(),
                'centers': centers[:take].detach(),
                'predicted_centers': centers[:take].detach(),
                'point_centers': centers[:take].detach(),
                'union_center_shift': (
                    union_center_shift[:take].detach()
                    if union_center_shift is not None else
                    centers[:take].new_zeros((take, 2))),
                'base_radius': debug_base_radius[:take].detach(),
                'support_radius': debug_support_radius[:take].detach(),
                'point_envelope': debug_support_radius[:take].detach(),
                'raw_expansion': debug_raw_expansion[:take].detach(),
                'expansion': debug_expansion[:take].detach(),
                'final_radius': debug_final_radius[:take].detach(),
                'base_coefficients': base_coefficients[:take].detach(),
                'refined_coefficients': refined_coefficients[:take].detach(),
                'blend': debug_blend[:take].detach(),
                'harmonic_gate': harmonic_gate[:take].detach(),
                'harmonic_gate_floor': gate_floor.detach(),
                'radial_softmax_beta': radial_softmax_beta,
                'fusion_mode': fusion_mode,
                'point_grad_scale': point_grad_scale,
                'point_mass': point_mass[:take].detach(),
                'reliability': reliability[:take].detach(),
                'roi_radius': roi_radius[:take].detach(),
            })

        geometry_store = GLOBAL_CANDIDATE_GEOMETRY_DICT.get(raw.device)
        if not isinstance(geometry_store, dict):
            geometry_store = {}
        resolution = (int(height), int(width))
        geometry_entry = geometry_store.get(resolution, {})
        geometry_entry.update({
            'support_prob': point_probability.detach(),
            'spectral_active_point': active_point.detach(),
            'spectral_soft_assignment': assignment_map.detach(),
            'candidate_prob': candidate_probability.detach(),
            'spectral_reliability': reliability_map.detach(),
            'spectral_harmonic_gate': gate_map.detach(),
            'spectral_expansion': expansion_map.detach(),
            'directions': angles.detach(),
            'topk_records': debug_records,
            'oracle_mode': False,
            'point_mode': 'candidate_conditioned_soft_spectral',
        })
        geometry_store[resolution] = geometry_entry
        GLOBAL_CANDIDATE_GEOMETRY_DICT[raw.device] = geometry_store
        stats = {
            'spectral_reliability': (
                torch.stack(reliability_values).mean()
                if reliability_values else work.new_zeros(())),
            'spectral_gate': (
                torch.stack(gate_values).mean()
                if gate_values else work.new_zeros(())),
        }
        return refined.reshape_as(work).to(dtype=raw.dtype), stats

    def forward(self, x):
        z = []


        self._discrete_reference_peak_cache = None
        point_mode = str(getattr(
            self, 'geometry_point_mode', 'soft')).strip().lower()
        native_point_logits = (
            GLOBAL_POINT_DICT.get(x[0].device)
            if point_mode == 'discrete' and len(x) else None)
        if torch.is_tensor(native_point_logits):
            target_batch = int(x[0].shape[0])
            if native_point_logits.shape[0] != target_batch:
                native_point_logits = (
                    native_point_logits[-target_batch:]
                    if native_point_logits.shape[0] >= target_batch else
                    None)
            if torch.is_tensor(native_point_logits):
                _, peak_threshold, peak_nms_kernel, _, _, _, _, _, _, _, _ =\
                    self._discrete_point_settings()
                native_probability = (
                    native_point_logits.detach().float().sigmoid())
                native_peaks = self._discrete_peak_support(
                    native_probability,
                    torch.ones_like(native_probability),
                    peak_threshold, peak_nms_kernel)
                self._discrete_reference_peak_cache = {
                    'map': native_peaks,
                    'resolution': (
                        int(native_peaks.shape[-2]),
                        int(native_peaks.shape[-1])),
                    'source': 'native_point_head',
                }
        temporal_adapters = getattr(self, 'temporal_adapters', ())
        for i in range(self.nl):
            if os.getenv('RKNN_model_hack', '0') != '0':
                z.append(torch.sigmoid(self.m[i](x[i])))
                continue
            feature = x[i]
            if (getattr(self, 'temporal_enabled', False) and
                    len(temporal_adapters) == self.nl):
                feature = temporal_adapters[i](feature)
            base_output = self.m[i](feature)
            oracle_base_store = GLOBAL_ORACLE_FOURIER_BASE_DICT.get(feature.device)
            oracle_base_feature = (
                oracle_base_store.get(tuple(feature.shape[-2:]))
                if isinstance(oracle_base_store, dict) else None)
            support_store = GLOBAL_FORMATION_SUPPORT_DICT.get(feature.device)
            support_entry = support_store.get(tuple(feature.shape[-2:])) if isinstance(support_store, dict) else None
            bs, _, ny, nx = base_output.shape
            raw = base_output.view(bs, self.na, self.ft_length, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            spectral_stats = None
            if bool(getattr(self, 'point_spectral_enabled', False)):
                raw, spectral_stats =\
                    self._candidate_conditioned_point_spectral_level(
                        raw, feature, i)
            if (torch.is_tensor(oracle_base_feature) and
                    oracle_base_feature.shape == feature.shape and
                    oracle_base_feature.data_ptr() != feature.data_ptr()):


                center_output = self.m[i](oracle_base_feature)
                center_raw = center_output.view(
                    bs, self.na, self.ft_length, ny, nx
                ).permute(0, 1, 3, 4, 2).contiguous()
                raw = torch.cat((center_raw[..., :2], raw[..., 2:]), dim=-1)
            support_probability = (
                support_entry.get('prob')
                if isinstance(support_entry, dict) else None)
            candidate_logits = (
                support_entry.get('candidate_logits')
                if isinstance(support_entry, dict) else None)
            center_override = None
            if isinstance(support_entry, dict):
                center_override = support_entry.get('center_override')
            oracle_geometry = self._oracle_geometry_level(raw, i)
            if isinstance(oracle_geometry, dict):
                support_probability = oracle_geometry['support_prob']
                candidate_logits = oracle_geometry['candidate_logits']
                center_override = oracle_geometry['center_override']
                if not isinstance(support_entry, dict):
                    support_entry = {}
                support_entry['oracle_geometry'] = True
                support_entry['prob'] = support_probability
                support_entry['candidate_logits'] = candidate_logits
            if (not bool(getattr(self, 'point_spectral_enabled', False)) and
                    torch.is_tensor(support_probability) and
                    torch.is_tensor(candidate_logits)):
                raw, geometry_stats = self._geometric_refine_level(
                    raw, support_probability, i,
                    candidate_logits=candidate_logits,
                    center_override=center_override)
                if isinstance(support_entry, dict):
                    support_entry['geometry_stats'] = geometry_stats
            elif (isinstance(support_entry, dict) and
                    isinstance(spectral_stats, dict)):
                support_entry['spectral_stats'] = spectral_stats
            x[i] = raw
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i],self.anchor_grid[i] = self._make_grid(nx, ny, i)


                y = x[i].float().clone()
                y[..., 0:2] = (self.grid[i] + 0.5 + y[..., 0:2]) * self.stride[i]

                b = y[..., 2:].shape[0]
                anchors_coef = self.anchor_grid[i].expand(b, -1, -1, -1, -1)
                y[..., 2:] = y[..., 2:]*anchors_coef

                if getattr(self, 'cm', None) is not None:
                    segments = y[..., 2:].reshape(*y.shape[:-1], -1, 4)
                    scale = self.ec + self.cm.to(y.device).view(1, 1, 1, 1, -1, 1)
                    y[..., 2:] = (segments * scale).reshape(*y.shape[:-1], -1)

                z.append(y)
        if os.getenv('RKNN_model_hack', '0') != '0':
            return z
        return x if self.training else (z, x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = next(self.parameters()).device
        if check_version(torch.__version__, '1.10.0'):
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])


        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()


        anchor_grid = (self.anchors[i].clone() * self.stride[i])\
            .view((1, self.na, 1, 1, 2, 1)).expand((1, self.na, ny, nx, 2, 2)).reshape(1, self.na, ny, nx, -1).repeat(1,1,1,1, self.ft_coef).float()
        return grid, anchor_grid

class P2PInfer(nn.Module):
    """
    ?(P2P Counting Head)
    ?(P3)?    """
    def __init__(self, c1, ch=(), stride=8):


        super().__init__()
        self.nc = c1
        self.stride = stride


        c_in = ch[0] if isinstance(ch, list) else ch


        self.stem = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_in),
            nn.SiLU(inplace=True)
        )


        self.cls_head = nn.Conv2d(c_in, self.nc, 1)
        self.reg_head = nn.Conv2d(c_in, 2, 1)
        self._initialize_biases()

    def forward(self, x):

        feat = x[0] if isinstance(x, list) else x


        reg_offsets = self.reg_head(feat)


        return {"pred_logits": cls_logits, "pred_offsets": reg_offsets}

    def _initialize_biases(self):

        b = self.cls_head.bias.view(-1)
        b.data.fill_(-4.59)
        self.cls_head.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class CausalPointTemporalAdapter(nn.Module):
    """Backbone-agnostic causal temporal refinement for point features.

    Input frames are clip-major flattened as ``[B*T, C, H, W]``.  Every
    output at time ``t`` depends only on frames ``0..t``.  A lightweight
    learned warp aligns the previous state to the current feature before a
    depthwise-separable ConvGRU update.  The adapter is residual, so the
    spatial point feature is never replaced by temporal memory.
    """

    def __init__(self, in_channels, state_channels=32, seq_len=5,
                 residual_gain=0.05, align=True, max_offset=2.0,
                 downsample=1, suppression_gain=0.0,
                 contrast_gain=0.0):
        super().__init__()
        state_channels = max(int(state_channels), 8)
        self.seq_len = max(int(seq_len), 1)
        self.align = bool(align)
        self.max_offset = max(float(max_offset), 0.0)
        self.downsample = max(int(downsample), 1)
        self.suppression_gain = min(max(float(suppression_gain), 0.0), 0.5)
        self.contrast_gain = min(max(float(contrast_gain), 0.0), 1.0)

        groups = min(8, state_channels)
        while state_channels % groups:
            groups -= 1
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, state_channels, 1, bias=False),
            nn.GroupNorm(groups, state_channels),
            nn.SiLU(inplace=True),
        )
        pair_channels = 2 * state_channels
        if self.align:
            self.offset_dw = nn.Conv2d(
                pair_channels, pair_channels, 3, padding=1,
                groups=pair_channels, bias=False)
            self.offset_pw = nn.Conv2d(pair_channels, 2, 1)
            nn.init.zeros_(self.offset_pw.weight)
            nn.init.zeros_(self.offset_pw.bias)

        self.gate_dw = nn.Conv2d(
            pair_channels, pair_channels, 3, padding=1,
            groups=pair_channels, bias=False)
        self.gate_pw = nn.Conv2d(pair_channels, 2 * state_channels, 1)
        self.candidate_dw = nn.Conv2d(
            pair_channels, pair_channels, 3, padding=1,
            groups=pair_channels, bias=False)
        self.candidate_pw = nn.Conv2d(pair_channels, state_channels, 1)
        self.out_proj = nn.Conv2d(state_channels, in_channels, 1, bias=False)
        evidence_channels = 3 * state_channels
        self.evidence_dw = nn.Conv2d(
            evidence_channels, evidence_channels, 3, padding=1,
            groups=evidence_channels, bias=False)
        self.evidence_pw = nn.Conv2d(evidence_channels, 2, 1)
        nn.init.zeros_(self.evidence_pw.weight)
        nn.init.constant_(self.evidence_pw.bias, -1.5)


        max_gain = 0.50
        init_ratio = min(max(float(residual_gain) / max_gain, 1e-4), 1.0 - 1e-4)
        self.max_gain = max_gain
        self.gain_logit = nn.Parameter(torch.tensor(
            math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))

    def _align_state(self, current, state):
        if not self.align or self.max_offset <= 0.0:
            return state
        offset_input = torch.cat((current, state), dim=1)
        offset = torch.tanh(self.offset_pw(F.silu(self.offset_dw(offset_input))))
        offset = offset * self.max_offset
        batch, _, height, width = state.shape
        y = torch.linspace(-1.0, 1.0, height, device=state.device, dtype=state.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=state.device, dtype=state.dtype)
        try:
            yy, xx = torch.meshgrid(y, x, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(y, x)
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        norm_x = 2.0 / max(width - 1, 1)
        norm_y = 2.0 / max(height - 1, 1)
        offset_grid = torch.stack(
            (offset[:, 0] * norm_x, offset[:, 1] * norm_y), dim=-1)
        return F.grid_sample(
            state, base_grid + offset_grid, mode='bilinear',
            padding_mode='border', align_corners=True)

    def _step(self, current, previous):
        previous = self._align_state(current, previous)
        pair = torch.cat((current, previous), dim=1)
        update, reset = torch.sigmoid(
            self.gate_pw(F.silu(self.gate_dw(pair)))).chunk(2, dim=1)
        candidate_pair = torch.cat((current, reset * previous), dim=1)
        candidate = torch.tanh(
            self.candidate_pw(F.silu(self.candidate_dw(candidate_pair))))
        return (1.0 - update) * previous + update * candidate

    def forward(self, feature):
        working = feature
        if self.downsample > 1:
            working = F.avg_pool2d(
                working, kernel_size=self.downsample,
                stride=self.downsample, ceil_mode=False)
        bt, _, height, width = working.shape
        seq_len = self.seq_len


        if seq_len <= 1 or bt < seq_len or bt % seq_len != 0:
            return feature
        batch = bt // seq_len
        projected = self.in_proj(working)
        projected = projected.view(
            batch, seq_len, projected.shape[1], height, width)
        state = torch.zeros_like(projected[:, 0])
        outputs = []
        enhance_masks = []
        suppress_masks = []
        for time_index in range(seq_len):
            current = projected[:, time_index]
            aligned_previous = self._align_state(current, state)
            difference = (current - aligned_previous).abs()
            state = self._step(current, state)
            if time_index == 0:


                outputs.append(torch.zeros_like(state))
                enhance_masks.append(torch.zeros_like(state[:, :1]))
                suppress_masks.append(torch.zeros_like(state[:, :1]))
                continue

            evidence_head = getattr(self, 'evidence_pw', None)
            if evidence_head is None:
                learned_enhance = torch.ones_like(state[:, :1])
                learned_suppress = torch.zeros_like(state[:, :1])
            else:
                evidence = torch.cat(
                    (current, aligned_previous, difference), dim=1)
                learned_enhance, learned_suppress = torch.sigmoid(
                    evidence_head(F.silu(self.evidence_dw(evidence)))).chunk(
                        2, dim=1)


            contrast = difference.mean(dim=1, keepdim=True)
            contrast_reference = contrast.mean(
                dim=(2, 3), keepdim=True).clamp_min(1e-6)
            motion = contrast / (contrast + contrast_reference)
            contrast_gain = float(getattr(self, 'contrast_gain', 0.0))
            enhance = (
                learned_enhance + contrast_gain * motion).clamp(0.0, 1.0)
            suppress = learned_suppress * (1.0 - motion)
            outputs.append(state)
            enhance_masks.append(enhance)
            suppress_masks.append(suppress)
        temporal = torch.stack(outputs, dim=1).reshape(
            bt, projected.shape[2], height, width)
        enhance_mask = torch.stack(enhance_masks, dim=1).reshape(
            bt, 1, height, width)
        suppress_mask = torch.stack(suppress_masks, dim=1).reshape(
            bt, 1, height, width)
        residual = self.out_proj(temporal)
        if residual.shape[-2:] != feature.shape[-2:]:
            residual = F.interpolate(
                residual, size=feature.shape[-2:], mode='bilinear',
                align_corners=False)
            enhance_mask = F.interpolate(
                enhance_mask, size=feature.shape[-2:], mode='bilinear',
                align_corners=False)
            suppress_mask = F.interpolate(
                suppress_mask, size=feature.shape[-2:], mode='bilinear',
                align_corners=False)
        gain = self.max_gain * torch.sigmoid(self.gain_logit).to(feature.dtype)
        suppression_gain = float(getattr(self, 'suppression_gain', 0.0))
        suppressed = feature * (
            1.0 - suppression_gain * suppress_mask.to(feature.dtype))
        return suppressed + gain * enhance_mask.to(feature.dtype) * F.relu(residual)


class ExplicitMotionPointTemporalAdapter(nn.Module):
    """Explicit causal motion gate for weak point-target features.

    Each frame is compared with an aligned median of all available history.
    A median-absolute-deviation scale prevents raw infrared flicker from being
    interpreted as motion, while local multi-frame support favors responses
    that form a short trajectory. Unlike the legacy positive-residual adapter,
    this module can strongly attenuate static/background features.

    Input and output remain clip-major ``[B*T, C, H, W]`` tensors. Frame zero
    is the exact spatial baseline because no temporal evidence exists yet.
    """

    def __init__(self, in_channels, state_channels=32, seq_len=5,
                 align=True, max_offset=2.0, downsample=2,
                 spatial_floor=0.15, enhance_gain=0.15,
                 motion_threshold=2.0, motion_temperature=0.5,
                 track_radius=2):
        super().__init__()
        state_channels = max(int(state_channels), 8)
        self.seq_len = max(int(seq_len), 1)
        self.align = bool(align)
        self.max_offset = max(float(max_offset), 0.0)
        self.downsample = max(int(downsample), 1)
        self.spatial_floor = min(max(float(spatial_floor), 0.05), 1.0)
        self.enhance_gain = max(float(enhance_gain), 0.0)
        self.motion_threshold = max(float(motion_threshold), 0.1)
        self.motion_temperature = max(float(motion_temperature), 0.05)
        self.track_radius = max(int(track_radius), 0)

        groups = min(8, state_channels)
        while state_channels % groups:
            groups -= 1
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, state_channels, 1, bias=False),
            nn.GroupNorm(groups, state_channels),
            nn.SiLU(inplace=True),
        )
        pair_channels = 2 * state_channels
        if self.align:
            self.offset_dw = nn.Conv2d(
                pair_channels, pair_channels, 3, padding=1,
                groups=pair_channels, bias=False)
            self.offset_pw = nn.Conv2d(pair_channels, 2, 1)
            nn.init.zeros_(self.offset_pw.weight)
            nn.init.zeros_(self.offset_pw.bias)

        evidence_channels = 4 * state_channels + 3
        hidden = max(state_channels, 16)
        self.gate_encoder = nn.Sequential(
            nn.Conv2d(evidence_channels, hidden, 1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.SiLU(inplace=True),
        )
        self.gate_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 1.0)
        self.motion_proj = nn.Sequential(
            nn.Conv2d(state_channels, state_channels, 3, padding=1,
                      groups=state_channels, bias=False),
            nn.GroupNorm(groups, state_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(state_channels, in_channels, 1, bias=False),
        )
        self.last_stats = {}


        self.last_debug = {}

    @staticmethod
    def _base_grid(batch, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        try:
            yy, xx = torch.meshgrid(y, x, indexing='ij')
        except TypeError:
            yy, xx = torch.meshgrid(y, x)
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(
            batch, -1, -1, -1)

    def _align_history(self, current, history):
        if not self.align or self.max_offset <= 0.0:
            return history
        pair = torch.cat((current, history), dim=1)
        offset = torch.tanh(
            self.offset_pw(F.silu(self.offset_dw(pair)))) * self.max_offset
        batch, _, height, width = history.shape
        grid = self._base_grid(
            batch, height, width, history.device, history.dtype)
        offset_grid = torch.stack((
            offset[:, 0] * (2.0 / max(width - 1, 1)),
            offset[:, 1] * (2.0 / max(height - 1, 1)),
        ), dim=-1)
        return F.grid_sample(
            history, grid + offset_grid, mode='bilinear',
            padding_mode='border', align_corners=True)

    def _trajectory_support(self, current, aligned_history):
        kernel = 2 * self.track_radius + 1
        current_contrast = F.relu(
            current - F.avg_pool2d(current, 5, stride=1, padding=2)
        ).mean(dim=1, keepdim=True)
        supports = []
        for history in aligned_history:
            contrast = F.relu(
                history - F.avg_pool2d(
                    history, 5, stride=1, padding=2)
            ).mean(dim=1, keepdim=True)
            contrast = contrast / contrast.mean(
                dim=(2, 3), keepdim=True).clamp_min(1e-4)
            support = torch.sigmoid(contrast - 1.0)
            if self.track_radius > 0:
                support = F.max_pool2d(
                    support, kernel, stride=1,
                    padding=self.track_radius)
            supports.append(support)
        if not supports:
            return torch.zeros_like(current_contrast)
        stack = torch.stack(supports, dim=1)
        if stack.shape[1] >= 2:

            strongest = stack.topk(2, dim=1).values
            support = torch.sqrt(
                (strongest[:, 0] * strongest[:, 1]).clamp_min(0.0))
        else:
            support = stack[:, 0]
        current_support = torch.sigmoid(
            current_contrast /
            current_contrast.mean(
                dim=(2, 3), keepdim=True).clamp_min(1e-4) - 1.0)
        return (support * current_support).clamp(0.0, 1.0)

    def forward(self, feature):
        working = feature
        if self.downsample > 1:
            working = F.avg_pool2d(
                working, kernel_size=self.downsample,
                stride=self.downsample)
        bt, _, height, width = working.shape
        seq_len = self.seq_len
        if seq_len <= 1 or bt < seq_len or bt % seq_len != 0:
            self.last_debug = {}
            return feature

        batch = bt // seq_len
        projected = self.in_proj(working).view(
            batch, seq_len, -1, height, width)
        outputs = []
        gate_logits_all = []
        gate_all = []
        residual_all = []
        positive_all = []
        snr_all = []
        instant_motion_all = []
        track_support_all = []
        learned_gate_all = []
        deterministic_gate_all = []
        history = []
        for time_index in range(seq_len):
            current = projected[:, time_index]
            if time_index == 0:
                outputs.append(feature.view(
                    batch, seq_len, *feature.shape[1:])[:, time_index])
                gate_logits_all.append(torch.zeros_like(current[:, :1]))
                gate_all.append(torch.ones_like(current[:, :1]))
                zero_map = torch.zeros_like(current[:, :1])
                residual_all.append(zero_map)
                positive_all.append(zero_map)
                snr_all.append(zero_map)
                instant_motion_all.append(zero_map)
                track_support_all.append(zero_map)
                learned_gate_all.append(torch.ones_like(zero_map))
                deterministic_gate_all.append(torch.ones_like(zero_map))
                history.append(current)
                continue

            aligned_history = [
                self._align_history(current, previous)
                for previous in history]
            history_stack = torch.stack(aligned_history, dim=1)
            background = history_stack.median(dim=1).values
            deviation = (
                history_stack - background[:, None]).abs()
            mad = deviation.median(dim=1).values
            residual = current - background
            positive = F.relu(residual)
            absolute = residual.abs()


            local_noise = F.avg_pool2d(
                absolute.mean(dim=1, keepdim=True),
                7, stride=1, padding=3)
            mad_energy = mad.mean(dim=1, keepdim=True)
            noise_floor = (
                mad_energy + 0.50 * local_noise +
                0.10 * local_noise.mean(
                    dim=(2, 3), keepdim=True)).clamp_min(1e-4)
            positive_energy = positive.mean(dim=1, keepdim=True)
            snr = (positive_energy / noise_floor).clamp(0.0, 20.0)
            instant_motion = torch.sigmoid(
                (snr - self.motion_threshold) /
                self.motion_temperature)
            track_support = self._trajectory_support(
                current, aligned_history)
            deterministic_gate = (
                instant_motion * (0.20 + 0.80 * track_support)
            ).clamp(1e-4, 1.0 - 1e-4)

            evidence = torch.cat((
                current, background, positive, mad,
                snr, instant_motion, track_support), dim=1)
            learned_gate = torch.sigmoid(
                self.gate_head(self.gate_encoder(evidence)))
            gate = (
                0.50 * learned_gate +
                0.50 * deterministic_gate).clamp(
                    1e-4, 1.0 - 1e-4)
            gate_logits = torch.logit(gate)

            gate_full = gate
            motion_residual = self.motion_proj(positive)
            if gate.shape[-2:] != feature.shape[-2:]:
                gate_full = F.interpolate(
                    gate, size=feature.shape[-2:],
                    mode='bilinear', align_corners=False)
                motion_residual = F.interpolate(
                    motion_residual, size=feature.shape[-2:],
                    mode='bilinear', align_corners=False)
            spatial = feature.view(
                batch, seq_len, *feature.shape[1:])[:, time_index]
            keep = self.spatial_floor + (
                1.0 - self.spatial_floor) * gate_full
            output = (
                keep.to(spatial.dtype) * spatial +
                self.enhance_gain * gate_full.to(spatial.dtype) *
                motion_residual)
            outputs.append(output)
            gate_logits_all.append(gate_logits)
            gate_all.append(gate)
            residual_all.append(absolute.mean(dim=1, keepdim=True))
            positive_all.append(positive_energy)
            snr_all.append(snr)
            instant_motion_all.append(instant_motion)
            track_support_all.append(track_support)
            learned_gate_all.append(learned_gate)
            deterministic_gate_all.append(deterministic_gate)
            history.append(current)

        output = torch.stack(outputs, dim=1).reshape_as(feature)
        gate_logits = torch.stack(gate_logits_all, dim=1).reshape(
            bt, 1, height, width)
        gates = torch.stack(gate_all, dim=1).reshape(
            bt, 1, height, width)
        GLOBAL_POINT_MOTION_DICT[feature.device] = {
            'gate_logits': gate_logits,
            'gate': gates,
            'seq_len': seq_len,
        }
        self.last_stats = {
            'gate_mean': gates.detach().mean(),
            'gate_max': gates.detach().amax(),
            'background_keep': (
                self.spatial_floor +
                (1.0 - self.spatial_floor) *
                gates.detach().mean()),
        }
        def _stack_debug(values):
            return torch.stack(values, dim=1).reshape(
                bt, 1, height, width).detach()

        self.last_debug = {
            'aligned_abs_residual': _stack_debug(residual_all),
            'positive_residual': _stack_debug(positive_all),
            'snr': _stack_debug(snr_all),
            'instant_motion': _stack_debug(instant_motion_all),
            'track_support': _stack_debug(track_support_all),
            'learned_gate': _stack_debug(learned_gate_all),
            'deterministic_gate': _stack_debug(deterministic_gate_all),
            'final_gate': gates.detach(),
            'gate_logits': gate_logits.detach(),
            'seq_len': seq_len,
            'batch': batch,
        }
        return output


class BioCountInfer(nn.Module):
    """
    Bio-inspired micro target point head.

    It predicts a center-response map and local offsets for dense micro targets.
    Counting is done only at inference time by counting response peaks inside the
    Fourier cluster contour, so this branch stays aligned with detection.
    """
    def __init__(self, nc=1, ch=(), stride=(), detach_features=False,
                 use_yolo_prior=False, yolo_box_scale=1.35,
                 yolo_prior_epsilon=0.20, yolo_conf=0.05, yolo_topk=20,
                 use_contour_features=True, yolo_prior_gain=0.50,
                 temporal_enabled=False, temporal_seq_len=5,
                 temporal_channels=32, temporal_gain=0.05,
                 temporal_align=True, temporal_max_offset=2.0,
                 out_scale=2, temporal_downsample=1,
                 temporal_suppress_gain=0.0,
                 temporal_contrast_gain=0.0,
                 yolo_outside_floor=1.0, yolo_logit_gain=0.0,
                 yolo_prior_softness=1.5, yolo_prior_nms_iou=1.0,
                 temporal_mode='gru', motion_spatial_floor=0.15,
                 motion_enhance_gain=0.15, motion_threshold=2.0,
                 motion_temperature=0.5, motion_track_radius=2,
                 motion_output_gate=False, motion_output_floor=0.02,
                 motion_output_gamma=2.0,
                 candidate_recall_enabled=False,
                 candidate_recall_conf=0.05,
                 candidate_recall_topk=4,
                 candidate_recall_anchor_scale=1.5,
                 candidate_recall_min_pixels=40.0,
                 candidate_recall_max_pixels=96.0,
                 candidate_recall_softness=0.20,
                 candidate_recall_max_gain=0.25,
                 candidate_recall_gain_init=-1.0,
                 fourier_to_point_enabled=False,
                 fourier_to_point_detach=True,
                 fourier_to_point_scale=1.60,
                 fourier_to_point_margin_pixels=6.0,
                 fourier_to_point_outside_floor=0.20,
                 fourier_to_point_gamma=1.0,
                 fourier_to_point_candidate_conf=0.20,
                 fourier_to_point_topk=4,
                 fourier_to_point_softness=0.20,
                 fourier_to_point_fallback_identity=True,
                 fourier_to_point_core_identity=False,
                 fourier_to_point_core_threshold=0.25,
                 fourier_to_point_density_rescue=False,
                 fourier_to_point_density_kernel=7,
                 fourier_to_point_density_min_mass=1.5,
                 fourier_to_point_density_temperature=0.25,
                 fourier_to_point_density_point_threshold=0.10,
                 fourier_to_point_rescue_max_distance=16.0,
                 fourier_to_point_rescue_gain=0.70):
        super().__init__()
        self.nc = nc
        self.detach_features = bool(detach_features)
        self.use_yolo_prior = bool(use_yolo_prior)
        self.yolo_box_scale = float(yolo_box_scale)
        self.yolo_prior_epsilon = float(yolo_prior_epsilon)
        self.yolo_conf = float(yolo_conf)
        self.yolo_topk = max(int(yolo_topk), 1)
        self.yolo_outside_floor = min(max(float(yolo_outside_floor), 0.0), 1.0)
        self.yolo_logit_gain = max(float(yolo_logit_gain), 0.0)
        self.yolo_prior_softness = max(float(yolo_prior_softness), 0.25)
        self.yolo_prior_nms_iou = min(max(float(yolo_prior_nms_iou), 0.0), 1.0)
        self.use_contour_features = bool(use_contour_features)


        self.yolo_prior_gain = max(float(yolo_prior_gain), 0.0)
        self.temporal_enabled = bool(temporal_enabled)
        self.temporal_mode = str(temporal_mode).strip().lower()
        self.motion_output_gate = bool(motion_output_gate)
        self.motion_output_floor = min(max(float(motion_output_floor), 0.0), 1.0)
        self.motion_output_gamma = max(float(motion_output_gamma), 0.0)


        self.candidate_recall_enabled = bool(candidate_recall_enabled)
        self.candidate_recall_conf = min(max(float(candidate_recall_conf), 0.0), 1.0)
        self.candidate_recall_topk = max(int(candidate_recall_topk), 1)
        self.candidate_recall_anchor_scale = max(float(candidate_recall_anchor_scale), 0.1)
        self.candidate_recall_min_pixels = max(float(candidate_recall_min_pixels), 1.0)
        self.candidate_recall_max_pixels = max(
            float(candidate_recall_max_pixels), self.candidate_recall_min_pixels)
        self.candidate_recall_softness = max(float(candidate_recall_softness), 1e-3)
        self.candidate_recall_max_gain = max(float(candidate_recall_max_gain), 0.0)


        self.fourier_to_point_enabled = bool(fourier_to_point_enabled)
        self.fourier_to_point_detach = bool(fourier_to_point_detach)
        self.fourier_to_point_scale = max(float(fourier_to_point_scale), 0.1)
        self.fourier_to_point_margin_pixels = max(
            float(fourier_to_point_margin_pixels), 0.0)
        self.fourier_to_point_outside_floor = min(max(
            float(fourier_to_point_outside_floor), 0.0), 1.0)
        self.fourier_to_point_gamma = max(float(fourier_to_point_gamma), 0.0)
        self.fourier_to_point_candidate_conf = min(max(
            float(fourier_to_point_candidate_conf), 0.0), 1.0)
        self.fourier_to_point_topk = max(int(fourier_to_point_topk), 1)
        self.fourier_to_point_softness = max(
            float(fourier_to_point_softness), 1e-3)
        self.fourier_to_point_fallback_identity = bool(
            fourier_to_point_fallback_identity)
        self.fourier_to_point_core_identity = bool(
            fourier_to_point_core_identity)
        self.fourier_to_point_core_threshold = min(max(
            float(fourier_to_point_core_threshold), 0.0), 1.0)
        self.fourier_to_point_density_rescue = bool(
            fourier_to_point_density_rescue)
        density_kernel = max(int(fourier_to_point_density_kernel), 1)
        self.fourier_to_point_density_kernel = (
            density_kernel if density_kernel % 2 else density_kernel + 1)
        self.fourier_to_point_density_min_mass = max(
            float(fourier_to_point_density_min_mass), 0.0)
        self.fourier_to_point_density_temperature = max(
            float(fourier_to_point_density_temperature), 1e-3)
        self.fourier_to_point_density_point_threshold = min(max(
            float(fourier_to_point_density_point_threshold), 0.0), 1.0)
        self.fourier_to_point_rescue_max_distance = max(
            float(fourier_to_point_rescue_max_distance), 0.0)
        self.fourier_to_point_rescue_gain = min(max(
            float(fourier_to_point_rescue_gain), 0.0), 1.0)
        self.seq_len = max(int(temporal_seq_len), 1)
        self.out_scale = max(int(out_scale), 1)
        self.stride = torch.tensor(stride, requires_grad=False) if len(stride) else torch.tensor([4.0])
        c2, c3, c4 = ch[:3]
        hidden = max(64, min(128, c2))

        self.p2_proj = nn.Sequential(nn.Conv2d(c2, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.p3_proj = nn.Sequential(nn.Conv2d(c3, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        self.p4_proj = nn.Sequential(nn.Conv2d(c4, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
        if self.candidate_recall_enabled:


            self.candidate_recall_proj = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1,
                          groups=hidden, bias=False),
                nn.GroupNorm(8, hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, hidden, 1, bias=False),
            )
            self.candidate_recall_gain_logit = nn.Parameter(
                torch.full((1,), float(candidate_recall_gain_init)))
        else:
            self.candidate_recall_proj = None
        if self.temporal_enabled and self.temporal_mode in {
                'motion_gate', 'explicit_motion_gate'}:
            self.temporal_adapter = ExplicitMotionPointTemporalAdapter(
                hidden, state_channels=temporal_channels,
                seq_len=self.seq_len, align=temporal_align,
                max_offset=temporal_max_offset,
                downsample=temporal_downsample,
                spatial_floor=motion_spatial_floor,
                enhance_gain=motion_enhance_gain,
                motion_threshold=motion_threshold,
                motion_temperature=motion_temperature,
                track_radius=motion_track_radius)
        elif self.temporal_enabled:
            self.temporal_adapter = CausalPointTemporalAdapter(
                hidden, state_channels=temporal_channels,
                seq_len=self.seq_len, residual_gain=temporal_gain,
                align=temporal_align, max_offset=temporal_max_offset,
                downsample=temporal_downsample,
                suppression_gain=temporal_suppress_gain,
                contrast_gain=temporal_contrast_gain)
        else:
            self.temporal_adapter = None

        self.on_filter = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.off_filter = nn.Conv2d(hidden, hidden, 7, padding=3, groups=hidden, bias=False)
        self.lateral = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.SiLU(inplace=True)
        )
        self.cls_head = nn.Conv2d(hidden // 2, nc, 1)
        self.reg_head = nn.Conv2d(hidden // 2, 2, 1)
        self.wh_head = nn.Conv2d(hidden // 2, 2, 1)


        self.macro_proj = None
        self.macro_proj_extra = nn.ModuleList()
        if len(ch) > 3:
            self.macro_proj = nn.Sequential(
                nn.Conv2d(ch[3], hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )
            for macro_channels in ch[4:]:
                self.macro_proj_extra.append(nn.Sequential(
                    nn.Conv2d(macro_channels, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, 1, 1),
                ))


            self.macro_gain = nn.Parameter(torch.full((1,), -2.0))

        self._initialize_biases()

    @torch.no_grad()
    def _build_yolo_prior(self, out_size, device, dtype):
        """Decode top macro predictions into an expanded soft candidate mask."""
        cached = GLOBAL_MACRO_DETECT_DICT.get(device)
        if not isinstance(cached, dict):
            return None
        raw_levels = cached.get('raw')
        anchors = cached.get('anchors')
        strides = cached.get('stride')
        if not isinstance(raw_levels, (list, tuple)) or not raw_levels:
            return None

        batch = raw_levels[0].shape[0]
        out_h, out_w = out_size
        prior = torch.zeros((batch, 1, out_h, out_w), device=device, dtype=torch.float32)
        yy = torch.arange(out_h, device=device, dtype=torch.float32).view(out_h, 1)
        xx = torch.arange(out_w, device=device, dtype=torch.float32).view(1, out_w)

        for batch_idx in range(batch):
            candidates = []
            for level, prediction in enumerate(raw_levels):
                pred = prediction[batch_idx].detach().float()
                score = pred[..., 4].sigmoid()
                if self.nc > 1 and pred.shape[-1] > 5:
                    score = score * pred[..., 5:].sigmoid().amax(dim=-1)
                level_k = min(self.yolo_topk, score.numel())
                values, indices = score.reshape(-1).topk(level_k)
                cells = pred.shape[1] * pred.shape[2]
                anchor_idx = torch.div(indices, cells, rounding_mode='floor')
                remain = indices % cells
                gy = torch.div(remain, pred.shape[2], rounding_mode='floor')
                gx = remain % pred.shape[2]
                selected = pred[anchor_idx, gy, gx]
                stride = float(strides[level])
                anchor_px = anchors[level, anchor_idx].float() * stride
                center = (selected[:, :2].sigmoid() * 2.0 - 0.5 +
                          torch.stack((gx, gy), dim=1).float()) * stride
                size = (selected[:, 2:4].sigmoid() * 2.0).pow(2) * anchor_px
                size = size * self.yolo_box_scale
                image_h = pred.shape[1] * stride
                image_w = pred.shape[2] * stride
                center[:, 0] *= out_w / image_w
                center[:, 1] *= out_h / image_h
                size[:, 0] *= out_w / image_w
                size[:, 1] *= out_h / image_h
                candidates.extend(zip(values, center, size))

            candidates.sort(key=lambda item: float(item[0]), reverse=True)


            selected_candidates = []
            for candidate in candidates:
                score, center, size = candidate
                half = size * 0.5
                box = torch.stack((center[0] - half[0], center[1] - half[1],
                                   center[0] + half[0], center[1] + half[1]))
                keep = True
                for _, _, _, kept_box in selected_candidates:
                    inter_lt = torch.maximum(box[:2], kept_box[:2])
                    inter_rb = torch.minimum(box[2:], kept_box[2:])
                    inter = (inter_rb - inter_lt).clamp_min(0.0).prod()
                    area_a = (box[2:] - box[:2]).clamp_min(0.0).prod()
                    area_b = (kept_box[2:] - kept_box[:2]).clamp_min(0.0).prod()
                    iou = inter / (area_a + area_b - inter).clamp_min(1e-6)
                    if float(iou) > self.yolo_prior_nms_iou:
                        keep = False
                        break
                if keep:
                    selected_candidates.append((score, center, size, box))
                if len(selected_candidates) >= self.yolo_topk:
                    break

            for score, center, size, _ in selected_candidates:
                score_weight = torch.sigmoid((score - self.yolo_conf) / 0.05)
                half_w, half_h = size[0] * 0.5, size[1] * 0.5
                softness = self.yolo_prior_softness
                mask_x = (torch.sigmoid((xx - (center[0] - half_w)) / softness) *
                          torch.sigmoid(((center[0] + half_w) - xx) / softness))
                mask_y = (torch.sigmoid((yy - (center[1] - half_h)) / softness) *
                          torch.sigmoid(((center[1] + half_h) - yy) / softness))
                prior[batch_idx, 0] = torch.maximum(
                    prior[batch_idx, 0], score_weight * mask_x * mask_y)
        return prior.to(dtype=dtype)

    @torch.no_grad()
    def _build_fourier_candidate_recall_prior(
            self, out_size, device, dtype):
        """Rasterize loose FourierProposal regions on the point feature grid.

        The proposal cache is detached by FourierProposal.  Only grid centers,
        anchor sizes and objectness are used here; the predicted Fourier curve
        is intentionally not used, so a contracted contour cannot remove a
        peripheral point from the point branch.
        """
        cached = GLOBAL_FOURIER_PROPOSAL_DICT.get(device)
        if not isinstance(cached, dict) or not cached:
            return None

        out_h, out_w = int(out_size[0]), int(out_size[1])
        prior = None
        yy = torch.arange(
            out_h, device=device, dtype=torch.float32).view(1, 1, out_h, 1)
        xx = torch.arange(
            out_w, device=device, dtype=torch.float32).view(1, 1, 1, out_w)

        for entry in cached.values():
            probability = entry.get('prob') if isinstance(entry, dict) else None
            anchor_pixels = entry.get('anchor_pixels') if isinstance(entry, dict) else None
            image_hw = entry.get('image_hw') if isinstance(entry, dict) else None
            if (not torch.is_tensor(probability) or probability.ndim != 5 or
                    not torch.is_tensor(anchor_pixels) or image_hw is None):
                continue

            probability = probability.detach().to(device=device, dtype=torch.float32)
            batch, anchors_n, height, width, _ = probability.shape
            score_map = probability[..., 0]
            level_k = min(int(getattr(
                self, 'candidate_recall_topk', 4)), score_map[0].numel())
            if level_k <= 0:
                continue
            scores, indices = score_map.reshape(batch, -1).topk(
                level_k, dim=1)
            cells = height * width
            anchor_idx = torch.div(indices, cells, rounding_mode='floor')
            remain = indices % cells
            grid_y = torch.div(remain, width, rounding_mode='floor')
            grid_x = remain % width

            image_h, image_w = float(image_hw[0]), float(image_hw[1])
            center_x = (grid_x.float() + 0.5) * (out_w / float(width))
            center_y = (grid_y.float() + 0.5) * (out_h / float(height))
            anchors = anchor_pixels.detach().to(
                device=device, dtype=torch.float32)
            selected_anchors = anchors[anchor_idx]
            scale = max(float(getattr(
                self, 'candidate_recall_anchor_scale', 1.5)), 0.1)
            min_radius = max(float(getattr(
                self, 'candidate_recall_min_pixels', 40.0)), 1.0)
            max_radius = max(float(getattr(
                self, 'candidate_recall_max_pixels', 96.0)), min_radius)
            radius_x_px = (0.5 * selected_anchors[..., 0] * scale).clamp(
                min=min_radius, max=max_radius)
            radius_y_px = (0.5 * selected_anchors[..., 1] * scale).clamp(
                min=min_radius, max=max_radius)
            radius_x = (radius_x_px * out_w / max(image_w, 1.0)).clamp_min(1.0)
            radius_y = (radius_y_px * out_h / max(image_h, 1.0)).clamp_min(1.0)

            distance = torch.sqrt(
                ((xx - center_x[..., None, None]) /
                 radius_x[..., None, None]).square() +
                ((yy - center_y[..., None, None]) /
                 radius_y[..., None, None]).square() + 1e-6)
            softness = max(float(getattr(
                self, 'candidate_recall_softness', 0.20)), 1e-3)
            soft_regions = torch.sigmoid((1.0 - distance) / softness)
            confidence = min(max(float(getattr(
                self, 'candidate_recall_conf', 0.05)), 0.0), 1.0)


            confidence_span = max(0.20 - confidence, 1e-3)
            confidence_gate = (
                (scores - confidence) / confidence_span).clamp(0.0, 1.0)
            level_prior = (
                soft_regions * confidence_gate[..., None, None]
            ).amax(dim=1, keepdim=True)
            prior = level_prior if prior is None else torch.maximum(
                prior, level_prior)

        if prior is None:
            return None
        return prior.clamp_(0.0, 1.0).to(dtype=dtype)

    @torch.no_grad()
    def _build_fourier_to_point_prior(self, out_size, device, dtype):
        """Rasterize expanded FourierProposal candidates for point filtering.

        This feedback is candidate-conditioned rather than contour-conditioned:
        FourierProposal runs before BioCountInfer, whereas the final Fourier
        coefficients are decoded after the point-to-Fourier bridge.  Using the
        coarse proposal here breaks that circular dependency and ensures the
        point-expanded final curve is never fed back as its own hard mask.
        """
        cached = GLOBAL_FOURIER_PROPOSAL_DICT.get(device)
        if not isinstance(cached, dict) or not cached:
            return None

        out_h, out_w = int(out_size[0]), int(out_size[1])
        prior = None
        yy = torch.arange(
            out_h, device=device, dtype=torch.float32).view(1, 1, out_h, 1)
        xx = torch.arange(
            out_w, device=device, dtype=torch.float32).view(1, 1, 1, out_w)

        for entry in cached.values():
            probability = entry.get('prob') if isinstance(entry, dict) else None
            anchor_pixels = entry.get('anchor_pixels') if isinstance(entry, dict) else None
            image_hw = entry.get('image_hw') if isinstance(entry, dict) else None
            if (not torch.is_tensor(probability) or probability.ndim != 5 or
                    not torch.is_tensor(anchor_pixels) or image_hw is None):
                continue

            probability = probability.detach().to(
                device=device, dtype=torch.float32)
            batch, _, height, width, _ = probability.shape
            score_map = probability[..., 0]
            level_k = min(int(getattr(
                self, 'fourier_to_point_topk', 4)), score_map[0].numel())
            if level_k <= 0:
                continue
            scores, indices = score_map.reshape(batch, -1).topk(
                level_k, dim=1)
            cells = height * width
            anchor_idx = torch.div(indices, cells, rounding_mode='floor')
            remain = indices % cells
            grid_y = torch.div(remain, width, rounding_mode='floor')
            grid_x = remain % width

            image_h, image_w = float(image_hw[0]), float(image_hw[1])
            center_x = (grid_x.float() + 0.5) * (out_w / float(width))
            center_y = (grid_y.float() + 0.5) * (out_h / float(height))
            anchors = anchor_pixels.detach().to(
                device=device, dtype=torch.float32)
            selected_anchors = anchors[anchor_idx]
            scale = max(float(getattr(
                self, 'fourier_to_point_scale', 1.60)), 0.1)
            margin = max(float(getattr(
                self, 'fourier_to_point_margin_pixels', 6.0)), 0.0)
            radius_x_px = 0.5 * selected_anchors[..., 0] * scale + margin
            radius_y_px = 0.5 * selected_anchors[..., 1] * scale + margin
            radius_x = (radius_x_px * out_w / max(image_w, 1.0)).clamp_min(1.0)
            radius_y = (radius_y_px * out_h / max(image_h, 1.0)).clamp_min(1.0)

            distance = torch.sqrt(
                ((xx - center_x[..., None, None]) /
                 radius_x[..., None, None]).square() +
                ((yy - center_y[..., None, None]) /
                 radius_y[..., None, None]).square() + 1e-6)
            softness = max(float(getattr(
                self, 'fourier_to_point_softness', 0.20)), 1e-3)
            soft_regions = torch.sigmoid((1.0 - distance) / softness)
            confidence = min(max(float(getattr(
                self, 'fourier_to_point_candidate_conf', 0.20)), 0.0), 1.0)


            confidence_gate = (scores >= confidence).to(soft_regions.dtype)
            level_prior = (
                soft_regions * confidence_gate[..., None, None]
            ).amax(dim=1, keepdim=True)
            prior = level_prior if prior is None else torch.maximum(
                prior, level_prior)

        if prior is None:
            return None
        return prior.clamp_(0.0, 1.0).to(dtype=dtype)

    def forward(self, x):
        p2, p3, p4 = x[:3]
        macro_features = list(x[3:])
        detach_features = getattr(self, 'detach_features', True)
        out_scale = getattr(self, 'out_scale', 2)
        if detach_features:
            p2, p3, p4 = p2.detach(), p3.detach(), p4.detach()
            macro_features = [feature.detach() for feature in macro_features]
        out_size = p2.shape[-2:]

        p2 = self.p2_proj(p2)
        p3 = F.interpolate(self.p3_proj(p3), size=out_size, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.p4_proj(p4), size=out_size, mode='bilinear', align_corners=False)


        feat = (p2 + p3 + p4) / 3.0
        temporal_adapter = getattr(self, 'temporal_adapter', None)
        if getattr(self, 'temporal_enabled', False) and temporal_adapter is not None:
            feat = temporal_adapter(feat)
        candidate_recall_prior = None
        candidate_recall_gain = feat.new_zeros(1)
        candidate_recall_proj = getattr(self, 'candidate_recall_proj', None)
        if (bool(getattr(self, 'candidate_recall_enabled', False)) and
                candidate_recall_proj is not None):
            candidate_recall_prior = self._build_fourier_candidate_recall_prior(
                out_size, feat.device, feat.dtype)
            if (torch.is_tensor(candidate_recall_prior) and
                    candidate_recall_prior.shape[0] == feat.shape[0]):
                gain_logit = getattr(self, 'candidate_recall_gain_logit', None)
                if gain_logit is None:
                    gain_logit = feat.new_tensor(-1.0)
                candidate_recall_gain = (
                    max(float(getattr(
                        self, 'candidate_recall_max_gain', 0.25)), 0.0) *
                    torch.sigmoid(gain_logit).to(feat.dtype))
                positive_residual = F.relu(candidate_recall_proj(feat))
                feat = feat + (
                    candidate_recall_gain * candidate_recall_prior *
                    positive_residual)
        macro_prior = None


        if getattr(self, 'use_yolo_prior', False):
            macro_prior = self._build_yolo_prior(out_size, feat.device, feat.dtype)
        contour_priors = []


        macro_proj = getattr(self, 'macro_proj', None)
        macro_proj_extra = getattr(self, 'macro_proj_extra', ())
        macro_projectors = ([macro_proj] if macro_proj is not None else []) + list(macro_proj_extra)
        if getattr(self, 'use_contour_features', True):
            for macro_feature, projector in zip(macro_features, macro_projectors):
                contour_priors.append(torch.sigmoid(F.interpolate(
                    projector(macro_feature), size=out_size,
                    mode='bilinear', align_corners=False)))
        if contour_priors:


            contour_union = 1.0 - torch.stack(
                [1.0 - prior for prior in contour_priors], dim=0).prod(dim=0)


            macro_gain = getattr(self, 'macro_gain', None)
            if macro_gain is None:
                macro_gain = contour_union.new_tensor(-2.0)
            contour_gain = torch.sigmoid(macro_gain).to(contour_union.dtype)
            contour_prior = 1.0 - contour_gain * (1.0 - contour_union)
            macro_prior = contour_prior if macro_prior is None else macro_prior * contour_prior
        if macro_prior is not None:
            prior_gain = max(float(getattr(self, 'yolo_prior_gain', 0.50)), 0.0)
            outside_floor = min(max(float(
                getattr(self, 'yolo_outside_floor', 1.0)), 0.0), 1.0)


            soft_gate = outside_floor + (
                1.0 + prior_gain - outside_floor) * macro_prior
            feat = feat * soft_gate
        center = self.on_filter(feat)
        surround = self.off_filter(feat)
        feat = F.silu(center - surround)

        inhibit = torch.sigmoid(self.lateral(feat))
        feat = feat * (1.0 - 0.5 * inhibit)
        feat = self.decoder(feat)
        if out_scale > 1:
            feat = F.interpolate(feat, scale_factor=out_scale, mode='bilinear', align_corners=False)

        if hasattr(self, "cls_head"):
            point_logits = self.cls_head(feat)
            offsets = self.reg_head(feat).sigmoid()
            wh = F.softplus(self.wh_head(feat)) if hasattr(self, "wh_head") else torch.zeros(
                (feat.shape[0], 2, feat.shape[2], feat.shape[3]), device=feat.device, dtype=feat.dtype)
        else:
            point_logits = self.peak_head(feat) if hasattr(self, "peak_head") else self.density_head(feat)
            offsets = torch.full((feat.shape[0], 2, feat.shape[2], feat.shape[3]),
                                 0.5, device=feat.device, dtype=feat.dtype)
            wh = torch.zeros((feat.shape[0], 2, feat.shape[2], feat.shape[3]),
                             device=feat.device, dtype=feat.dtype)
        if macro_prior is not None:
            prior_out = F.interpolate(
                macro_prior, size=point_logits.shape[-2:],
                mode='bilinear', align_corners=False)
            logit_gain = max(float(getattr(self, 'yolo_logit_gain', 0.0)), 0.0)
            if logit_gain > 0.0:


                point_logits = point_logits + logit_gain * (2.0 * prior_out - 1.0)


        motion_output_gate = None
        motion_mode = str(getattr(self, 'temporal_mode', '')).strip().lower()
        if (getattr(self, 'temporal_enabled', False) and
                motion_mode in {'motion_gate', 'explicit_motion_gate'} and
                bool(getattr(self, 'motion_output_gate', False))):
            motion_state = GLOBAL_POINT_MOTION_DICT.get(point_logits.device)
            gate = motion_state.get('gate') if isinstance(motion_state, dict) else None
            if (torch.is_tensor(gate) and gate.ndim == 4 and
                    gate.shape[0] == point_logits.shape[0]):
                if gate.shape[-2:] != point_logits.shape[-2:]:
                    gate = F.interpolate(
                        gate, size=point_logits.shape[-2:],
                        mode='bilinear', align_corners=False)
                output_floor = min(max(float(
                    getattr(self, 'motion_output_floor', 0.02)), 0.0), 1.0)
                output_gamma = max(float(
                    getattr(self, 'motion_output_gamma', 2.0)), 0.0)
                motion_output_gate = (
                    output_floor + (1.0 - output_floor) * gate.float()
                ).pow(output_gamma).clamp(1e-4, 1.0)
                point_dtype = point_logits.dtype
                raw_point_logits = point_logits.float()


                log_gated_probability = (
                    F.logsigmoid(raw_point_logits) +
                    torch.log(motion_output_gate))
                gated_probability = torch.exp(log_gated_probability)
                gated_point_logits = (
                    log_gated_probability -
                    torch.log1p(-gated_probability.clamp(
                        max=1.0 - torch.finfo(torch.float32).eps)))


                point_logits = torch.where(
                    motion_output_gate >= 1.0 - 1e-7,
                    raw_point_logits,
                    gated_point_logits).to(point_dtype)
                motion_state['output_gate'] = motion_output_gate
                temporal_adapter = getattr(self, 'temporal_adapter', None)
                if temporal_adapter is not None and hasattr(
                        temporal_adapter, 'last_stats'):
                    temporal_adapter.last_stats.update({
                        'output_gate_mean': motion_output_gate.detach().mean(),
                        'output_gate_min': motion_output_gate.detach().amin(),
                        'output_gate_max': motion_output_gate.detach().amax(),
                    })


        raw_point_logits = point_logits
        fourier_point_prior = None
        fourier_point_gate = None
        fourier_point_core = None
        fourier_point_density = None
        fourier_point_rescue = None
        if bool(getattr(self, 'fourier_to_point_enabled', False)):
            fourier_point_prior = self._build_fourier_to_point_prior(
                point_logits.shape[-2:], point_logits.device,
                point_logits.dtype)
            if (torch.is_tensor(fourier_point_prior) and
                    fourier_point_prior.shape[0] == point_logits.shape[0]):
                if bool(getattr(self, 'fourier_to_point_detach', True)):
                    fourier_point_prior = fourier_point_prior.detach()
                outside_floor = min(max(float(getattr(
                    self, 'fourier_to_point_outside_floor', 0.20)), 0.0), 1.0)
                gate_gamma = max(float(getattr(
                    self, 'fourier_to_point_gamma', 1.0)), 0.0)
                fourier_point_gate = (
                    outside_floor +
                    (1.0 - outside_floor) * fourier_point_prior.float()
                ).pow(gate_gamma).clamp(1e-4, 1.0)


                if bool(getattr(
                        self, 'fourier_to_point_core_identity', False)):
                    core_threshold = min(max(float(getattr(
                        self, 'fourier_to_point_core_threshold', 0.25)),
                        0.0), 1.0)
                    fourier_point_core = (
                        fourier_point_prior.float() >= core_threshold)
                    fourier_point_gate = torch.where(
                        fourier_point_core, torch.ones_like(
                            fourier_point_gate), fourier_point_gate)


                if bool(getattr(
                        self, 'fourier_to_point_density_rescue', False)):
                    density_kernel = max(int(getattr(
                        self, 'fourier_to_point_density_kernel', 7)), 1)
                    if density_kernel % 2 == 0:
                        density_kernel += 1
                    point_threshold = min(max(float(getattr(
                        self, 'fourier_to_point_density_point_threshold',
                        0.10)), 0.0), 1.0)
                    raw_probability = raw_point_logits.float().sigmoid()
                    positive_evidence = (
                        (raw_probability - point_threshold).clamp_min(0.0) /
                        max(1.0 - point_threshold, 1e-3))
                    local_mass = F.avg_pool2d(
                        positive_evidence, kernel_size=density_kernel,
                        stride=1, padding=density_kernel // 2
                    ) * float(density_kernel * density_kernel)
                    min_mass = max(float(getattr(
                        self, 'fourier_to_point_density_min_mass', 1.5)),
                        0.0)
                    mass_temperature = max(float(getattr(
                        self, 'fourier_to_point_density_temperature', 0.25)),
                        1e-3)
                    fourier_point_density = torch.sigmoid(
                        (local_mass - min_mass) / mass_temperature)

                    point_stride = max(float(
                        self.stride[0].item() if self.stride.numel() else 2.0
                    ) / max(float(getattr(self, 'out_scale', 1)), 1.0), 1.0)
                    rescue_pixels = max(float(getattr(
                        self, 'fourier_to_point_rescue_max_distance', 16.0)),
                        0.0)
                    rescue_cells = max(int(round(
                        rescue_pixels / point_stride)), 0)
                    halo_seed_threshold = min(max(float(getattr(
                        self, 'fourier_to_point_core_threshold', 0.25)),
                        0.0), 1.0)
                    accepted = (
                        fourier_point_prior.float() >=
                        halo_seed_threshold).float()
                    if rescue_cells > 0:
                        accepted_halo = F.max_pool2d(
                            accepted, kernel_size=2 * rescue_cells + 1,
                            stride=1, padding=rescue_cells)
                    else:
                        accepted_halo = accepted
                    density_rescue = (
                        fourier_point_density * accepted_halo).clamp(0.0, 1.0)
                    rescue_gain = min(max(float(getattr(
                        self, 'fourier_to_point_rescue_gain', 0.70)),
                        0.0), 1.0)
                    rescue_gate = (
                        outside_floor +
                        (rescue_gain - outside_floor) * density_rescue)
                    fourier_point_gate = torch.maximum(
                        fourier_point_gate, rescue_gate)
                    fourier_point_rescue = density_rescue

                if bool(getattr(
                        self, 'fourier_to_point_fallback_identity', True)):
                    valid = fourier_point_prior.float().flatten(1).amax(
                        dim=1) > 1e-6
                    fourier_point_gate = torch.where(
                        valid[:, None, None, None], fourier_point_gate,
                        torch.ones_like(fourier_point_gate))

                point_dtype = point_logits.dtype
                point_logits_f32 = point_logits.float()
                log_guided_probability = (
                    F.logsigmoid(point_logits_f32) +
                    torch.log(fourier_point_gate))
                guided_probability = torch.exp(log_guided_probability)
                guided_logits = (
                    log_guided_probability -
                    torch.log1p(-guided_probability.clamp(
                        max=1.0 - torch.finfo(torch.float32).eps)))
                point_logits = torch.where(
                    fourier_point_gate >= 1.0 - 1e-7,
                    point_logits_f32, guided_logits).to(point_dtype)

        support_logits = point_logits
        if macro_prior is not None:
            prior_out = F.interpolate(
                macro_prior, size=point_logits.shape[-2:], mode='bilinear', align_corners=False)
            prior_gain = max(float(getattr(self, 'yolo_prior_gain', 0.50)), 0.0)
            point_probability = point_logits.sigmoid()


            boost = (prior_gain * prior_out).clamp(0.0, 1.0)
            boosted_probability = (
                1.0 - (1.0 - point_probability) * (1.0 - boost)
            ).clamp(1e-5, 1.0 - 1e-5)
            support_logits = torch.logit(boosted_probability)


        GLOBAL_POINT_DICT[point_logits.device] = support_logits
        GLOBAL_POINT_FEATURE_DICT[point_logits.device] = {
            'feature': feat,
            'raw_point_logits': raw_point_logits,
            'support_logits': support_logits,
            'macro_prior': macro_prior,
            'candidate_recall_prior': candidate_recall_prior,
            'candidate_recall_gain': candidate_recall_gain.detach(),
            'fourier_point_prior': fourier_point_prior,
            'fourier_point_gate': fourier_point_gate,
            'fourier_point_core': fourier_point_core,
            'fourier_point_density': fourier_point_density,
            'fourier_point_rescue': fourier_point_rescue,
        }


        supervised_point_logits = (
            raw_point_logits if self.training else point_logits)
        return {
            "pred_logits": supervised_point_logits,
            "raw_pred_logits": raw_point_logits,
            "guided_pred_logits": point_logits,
            "pred_offsets": offsets,
            "pred_wh": wh,
            "macro_prior": macro_prior,
            "support_logits": support_logits,
            "motion_output_gate": motion_output_gate,
            "candidate_recall_prior": candidate_recall_prior,
            "candidate_recall_gain": candidate_recall_gain.detach(),
            "fourier_point_prior": fourier_point_prior,
            "fourier_point_gate": fourier_point_gate,
            "fourier_point_core": fourier_point_core,
            "fourier_point_density": fourier_point_density,
            "fourier_point_rescue": fourier_point_rescue,
        }

    def _initialize_biases(self):
        nn.init.constant_(self.cls_head.bias, -4.59)
        nn.init.constant_(self.reg_head.bias, 0.0)
        nn.init.constant_(self.wh_head.bias, 0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/repvit_m0_9_t2.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')

    opt = parser.parse_args()

    opt.cfg = check_yaml(opt.cfg)
    print_args(FILE.stem, opt)
    device = select_device(opt.device)


    model = Model(opt.cfg).to(device)


    summary(model, (3, 640, 640), batch_size=1, device="cuda")

    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)


    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')


