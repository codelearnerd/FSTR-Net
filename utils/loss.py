
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from copy import deepcopy
from utils.metrics import bbox_iou, ab_iou, wh_iou
from utils.torch_utils import is_parallel
from utils.general import pts2dir
import math
from hungarian.hungarian_match import hungarian_match
from utils.ft_utils import ft_correct_clockwise,solve_cos_sin_closed_form,angle_multiple,rotate_high_orders
from scipy.optimize import linear_sum_assignment
from models.common import (
    GLOBAL_MASK_DICT, GLOBAL_FORMATION_SUPPORT_DICT,
    GLOBAL_POINT_MOTION_DICT)

criterion_p = nn.CrossEntropyLoss()

def smooth_BCE(eps=0.1):

    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true

        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)


        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def bce_loss_without_sigmond(p, t):
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p))

def loop_loss(n_loop,ft_coefs_labels,ft_coef_pi,cm=None,ec=0.01):
    if n_loop==0:
        return ft_coefs_labels


    nt = ft_coefs_labels.shape[0]

    a_ft, b_ft, c_ft, d_ft = [coef.squeeze(-1) for coef in torch.split(ft_coefs_labels.view(nt, -1, 4), 1, dim = -1)]

    ft_coef_loop = torch.zeros([n_loop, nt, ft_coefs_labels.shape[1]]).to(ft_coef_pi.device)

    for t in range(n_loop):
        coef_num = a_ft.shape[-1]
        angle = torch.tensor(2*math.pi*t/n_loop)
        for k in range(coef_num):
            k_angle = (k+1) * angle
            t_cos = torch.cos(k_angle)
            t_sin = torch.sin(k_angle)


            ft_coef_loop[t, :, 4*k + 0] = t_cos * a_ft[:, k] + t_sin * b_ft[:, k]

            ft_coef_loop[t, :, 4*k + 1] =-t_sin * a_ft[:, k] + t_cos * b_ft[:, k]


            ft_coef_loop[t, :, 4*k + 2] = t_cos * c_ft[:, k] + t_sin * d_ft[:, k]

            ft_coef_loop[t, :, 4*k + 3] =-t_sin * c_ft[:, k] + t_cos * d_ft[:, k]


    loss_nt = torch.abs(ft_coef_loop - ft_coef_pi)
    if cm is not None:

        term = cm.shape[0]

        loss_nt = loss_nt.view(*loss_nt.shape[:-1], term, 4)

        loss_nt = loss_nt / (ec + cm.view(1, 1, term, 1))

        loss_nt = loss_nt.view(*loss_nt.shape[:-2], term*4)
    loss_nt = loss_nt.mean(-1)
    nt_indices = torch.argmin(loss_nt.T, dim=-1)
    ft_coef_min = ft_coef_loop[nt_indices, torch.arange(ft_coefs_labels.shape[0])]

    return ft_coef_min


def sample_fourier_curve(centers, coefficients, n_points=96):
    """Differentiably sample closed Fourier curves in the current feature-map coordinates."""
    terms = coefficients.shape[-1] // 4
    if terms <= 0:
        return centers.new_zeros((centers.shape[0], 0, 2))

    dtype = centers.dtype
    device = centers.device
    t = torch.linspace(0.0, 2.0 * math.pi, n_points + 1, device=device, dtype=dtype)[:-1]
    harmonics = torch.arange(1, terms + 1, device=device, dtype=dtype).view(1, terms, 1)
    angles = harmonics * t.view(1, 1, -1)
    cos_t, sin_t = angles.cos(), angles.sin()

    coeffs = coefficients.view(-1, terms, 4)
    x = centers[:, 0:1] + (coeffs[..., 0:1] * cos_t + coeffs[..., 1:2] * sin_t).sum(1)
    y = centers[:, 1:2] + (coeffs[..., 2:3] * cos_t + coeffs[..., 3:4] * sin_t).sum(1)
    return torch.stack((x, y), dim=-1)


def rasterize_fourier_macro_mask(macro_targets, batch_size, height, width, device, dtype, n_points=128):
    """Rasterize the Fourier formation GT into a non-differentiable supervision mask."""
    mask_np = np.zeros((batch_size, height, width), dtype=np.float32)
    if macro_targets.numel() == 0:
        return torch.from_numpy(mask_np).to(device=device, dtype=dtype).unsqueeze(1)

    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    for target in macro_targets:
        batch_index = int(target[0].item())
        if batch_index < 0 or batch_index >= batch_size:
            continue
        terms = max((target.shape[0] - 8) // 4, 0)
        center_x = target[6].item() * width
        center_y = target[7].item() * height
        if terms:
            coefs = target[8:8 + 4 * terms].detach().cpu().numpy()
            harmonic = np.arange(1, terms + 1, dtype=np.float32)[:, None]
            x_points = center_x + np.dot(coefs[0::4] * width, np.cos(harmonic * t)) + np.dot(
                coefs[1::4] * width, np.sin(harmonic * t))
            y_points = center_y + np.dot(coefs[2::4] * height, np.cos(harmonic * t)) + np.dot(
                coefs[3::4] * height, np.sin(harmonic * t))
        else:
            box_w, box_h = target[4].item() * width, target[5].item() * height
            x_points = np.array([center_x - box_w / 2, center_x + box_w / 2,
                                 center_x + box_w / 2, center_x - box_w / 2])
            y_points = np.array([center_y - box_h / 2, center_y - box_h / 2,
                                 center_y + box_h / 2, center_y + box_h / 2])
        polygon = np.stack((x_points, y_points), axis=1).astype(np.int32)
        cv2.fillPoly(mask_np[batch_index], [polygon], 1.0)
    return torch.from_numpy(mask_np).to(device=device, dtype=dtype).unsqueeze(1)


def select_last_sequence_support(support_logits, target_batch_size):
    """Align support logits from flattened video clips with last-frame targets."""
    if support_logits is None or support_logits.shape[0] == target_batch_size:
        return support_logits
    if target_batch_size > 0 and support_logits.shape[0] % target_batch_size == 0:
        seq_len = support_logits.shape[0] // target_batch_size
        return support_logits[seq_len - 1::seq_len]
    if target_batch_size == 1:
        return support_logits[-1:]
    return None


def get_support_level(support_store, grid_size, target_batch_size):
    """Return sequence-aligned support logits for one Fourier pyramid level."""
    if not isinstance(support_store, dict):
        return select_last_sequence_support(support_store, target_batch_size)
    entry = support_store.get((int(grid_size[0]), int(grid_size[1])))
    if not isinstance(entry, dict):
        return None
    return select_last_sequence_support(entry.get('logits'), target_batch_size)


def fourier_support_consistency_loss(pred_center, pred_coefficients, target_center,
                                     matched_batch, macro_targets, support_logits,
                                     grid_size, n_directions=32,
                                     support_threshold=0.10,
                                     undercoverage_gain=2.0,
                                     boundary_gain=1.0,
                                     preserve_peaks=False):
    """Align a decoded Fourier boundary with a point-derived support envelope.

    The support map is detached here and acts as a teacher for the Fourier
    contour. It is trained separately by the macro support loss, while its input
    point heatmap remains detached from all macro objectives.
    """
    zero = pred_center.sum() * 0.0
    if (pred_center.numel() == 0 or pred_coefficients.shape[-1] < 4 or
            support_logits is None or macro_targets.numel() == 0):
        return zero

    height, width = int(grid_size[0]), int(grid_size[1])
    if support_logits.shape[-2:] != (height, width):
        if (preserve_peaks and support_logits.shape[-2] >= height and
                support_logits.shape[-1] >= width):
            support_logits = F.adaptive_max_pool2d(
                support_logits, output_size=(height, width))
        else:
            support_logits = F.interpolate(
                support_logits, size=(height, width), mode='bilinear', align_corners=False)
    support_prob = support_logits.sigmoid().float().detach()

    n_directions = max(int(n_directions), 8)
    theta = torch.linspace(
        0.0, 2.0 * math.pi, n_directions + 1,
        device=pred_center.device, dtype=torch.float32)[:-1]
    directions = torch.stack((theta.cos(), theta.sin()), dim=1)
    pred_curve = sample_fourier_curve(
        pred_center.float(), pred_coefficients.float(),
        n_points=max(96, n_directions * 4))

    yy, xx = torch.meshgrid(
        torch.arange(height, device=pred_center.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=pred_center.device, dtype=torch.float32) + 0.5,
        indexing='ij')
    pixel_xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    macro_batches = macro_targets[:, 0].long()
    macro_centers = macro_targets[:, 6:8].float() * pred_center.new_tensor([width, height])
    losses = []

    for match_idx in range(pred_center.shape[0]):
        batch_idx = int(matched_batch[match_idx].item())
        if batch_idx < 0 or batch_idx >= support_prob.shape[0]:
            continue
        macro_ids = torch.where(macro_batches == batch_idx)[0]
        if macro_ids.numel() == 0:
            continue
        macro_local = torch.cdist(
            target_center[match_idx:match_idx + 1].float(),
            macro_centers[macro_ids]).argmin(dim=1)[0]
        macro_id = macro_ids[macro_local]
        box_wh = macro_targets[macro_id, 4:6].float() * pred_center.new_tensor([width, height])
        half_window = torch.maximum(box_wh * 0.75, box_wh.new_tensor([3.0, 3.0]))

        reference_center = target_center[match_idx].float()
        pixel_vectors = pixel_xy - reference_center
        local_mask = (
            (pixel_vectors[:, 0].abs() <= half_window[0]) &
            (pixel_vectors[:, 1].abs() <= half_window[1]))
        probability = F.relu(
            support_prob[batch_idx, 0].reshape(-1) - float(support_threshold))
        probability = probability * local_mask.to(probability.dtype)
        if probability.sum() <= 1e-6:
            continue

        pixel_radius = torch.linalg.vector_norm(pixel_vectors, dim=1).clamp_min(1e-6)
        pixel_direction = pixel_vectors / pixel_radius[:, None]
        diagonal = torch.linalg.vector_norm(box_wh).clamp_min(2.0)
        angular_score = torch.matmul(directions, pixel_direction.T)
        support_score = (
            probability.clamp_min(1e-8).log()[None, :] +
            14.0 * (angular_score - 1.0) +
            6.0 * (pixel_radius / diagonal)[None, :])
        support_weight = support_score.softmax(dim=1)
        support_projection = torch.matmul(directions, pixel_vectors.T).clamp_min(0.0)
        support_reach = (support_weight * support_projection).sum(dim=1)

        curve_vectors = pred_curve[match_idx] - reference_center
        curve_radius = torch.linalg.vector_norm(curve_vectors, dim=1).clamp_min(1e-6)
        curve_direction = curve_vectors / curve_radius[:, None]
        curve_angular = torch.matmul(directions, curve_direction.T)
        curve_score = 20.0 * (curve_angular - 1.0) + 4.0 * (curve_radius / diagonal)[None, :]
        curve_weight = curve_score.softmax(dim=1)
        curve_projection = torch.matmul(directions, curve_vectors.T).clamp_min(0.0)
        curve_reach = (curve_weight * curve_projection).sum(dim=1)

        normalized_pred = curve_reach / diagonal
        normalized_support = support_reach / diagonal
        boundary = F.smooth_l1_loss(normalized_pred, normalized_support)
        undercoverage = F.relu(normalized_support - normalized_pred).mean()
        losses.append(
            float(boundary_gain) * boundary +
            float(undercoverage_gain) * undercoverage)

    return torch.stack(losses).mean() if losses else zero


def fourier_curve_geometry_loss(pred_center, pred_coefficients, target_center, target_coefficients, n_points=96):
    """Return phase-invariant boundary and area losses for Fourier contours."""
    pred_curve = sample_fourier_curve(pred_center.float(), pred_coefficients.float(), n_points)
    target_curve = sample_fourier_curve(target_center.float(), target_coefficients.float(), n_points)
    if pred_curve.shape[1] == 0:
        zero = pred_center.sum() * 0.0
        return zero, zero


    pairwise_dist = torch.cdist(pred_curve, target_curve, p=2)
    pred_to_target = pairwise_dist.min(dim=2).values.mean(dim=1)
    target_to_pred = pairwise_dist.min(dim=1).values.mean(dim=1)
    target_extent = target_curve.amax(dim=1) - target_curve.amin(dim=1)
    target_diag = torch.linalg.vector_norm(target_extent, dim=1).clamp_min(1.0)
    curve_loss = ((pred_to_target + target_to_pred) * 0.5 / target_diag).mean()


    def polygon_area(curve):
        next_curve = torch.roll(curve, shifts=-1, dims=1)
        cross = curve[..., 0] * next_curve[..., 1] - curve[..., 1] * next_curve[..., 0]
        return 0.5 * cross.sum(dim=1).abs()

    pred_area = polygon_area(pred_curve).clamp_min(1e-4)
    target_area = polygon_area(target_curve).clamp_min(1e-4)
    area_loss = F.smooth_l1_loss(pred_area.log(), target_area.log())
    return curve_loss, area_loss


def fourier_normalized_shape_loss(pred_center, pred_coefficients, target_center,
                                  target_coefficients, n_points=96):
    """Compare contour shape after removing translation and global radial scale.

    Strong coverage is cheaply minimized by inflating the first harmonic into an
    almost circular envelope. RMS-radius normalization makes this term supervise
    angular deformation and high-order harmonics without requesting contraction.
    """
    pred_curve = sample_fourier_curve(
        pred_center.float(), pred_coefficients.float(), n_points)
    target_curve = sample_fourier_curve(
        target_center.float(), target_coefficients.float(), n_points)
    if pred_curve.shape[1] == 0:
        return pred_center.sum() * 0.0

    pred_offset = pred_curve - pred_center.float()[:, None, :]
    target_offset = target_curve - target_center.float()[:, None, :]
    pred_scale = torch.sqrt(
        pred_offset.square().sum(dim=2).mean(dim=1).clamp_min(1e-6))
    target_scale = torch.sqrt(
        target_offset.square().sum(dim=2).mean(dim=1).clamp_min(1e-6))
    pred_shape = pred_offset / pred_scale[:, None, None]
    target_shape = target_offset / target_scale[:, None, None]

    pairwise_dist = torch.cdist(pred_shape, target_shape, p=2)
    pred_to_target = pairwise_dist.min(dim=2).values.mean(dim=1)
    target_to_pred = pairwise_dist.min(dim=1).values.mean(dim=1)
    return (0.5 * (pred_to_target + target_to_pred)).mean()


def fourier_harmonic_shape_loss(pred_coefficients, target_coefficients):
    """Match relative Fourier harmonics without letting the first order dominate.

    Coefficients are normalized by each contour's first-harmonic energy, so this
    term does not determine global contour size. A mild sqrt(order) weighting
    prevents the many small high-order targets from being ignored by the large
    first harmonic.
    """
    terms = pred_coefficients.shape[-1] // 4
    if terms <= 0:
        return pred_coefficients.sum() * 0.0
    pred = pred_coefficients.float().view(-1, terms, 4)
    target = target_coefficients.float().view(-1, terms, 4)
    pred_scale = torch.sqrt(pred[:, 0].square().mean(dim=1).clamp_min(1e-6))
    target_scale = torch.sqrt(target[:, 0].square().mean(dim=1).clamp_min(1e-6))
    pred_relative = pred / pred_scale[:, None, None]
    target_relative = target / target_scale[:, None, None]
    order_weight = torch.arange(
        1, terms + 1, device=pred.device, dtype=pred.dtype).sqrt()
    order_weight = order_weight / order_weight.mean()
    error = F.smooth_l1_loss(
        pred_relative, target_relative, reduction='none').mean(dim=2)
    return (error * order_weight[None, :]).mean()


def fourier_directional_support_loss(
        pred_center, pred_coefficients, target_center, target_coefficients,
        grid_size, n_directions=64, under_gain=2.0, over_gain=0.5,
        radial_band=0.00625):
    """Asymmetric full-angle support loss for the Fourier envelope.

    Point coverage constrains only directions containing micro targets, leaving
    empty directions free to expand as part of a first-harmonic circle. This loss
    evaluates the support function of the predicted and native GT contours over
    the full angular domain. Coordinates are normalized by the feature-grid size,
    so ``radial_band`` is expressed as a fraction of the input image size.
    """
    zero = pred_center.sum() * 0.0
    if pred_center.numel() == 0 or pred_coefficients.shape[-1] < 4:
        return zero

    height, width = int(grid_size[0]), int(grid_size[1])
    scale_xy = pred_center.new_tensor([width, height])
    terms = pred_coefficients.shape[-1] // 4
    coefficient_scale = pred_coefficients.new_tensor(
        [width, width, height, height]).repeat(terms)
    pred_center_norm = pred_center.float() / scale_xy
    target_center_norm = target_center.float() / scale_xy
    pred_curve = sample_fourier_curve(
        pred_center_norm, pred_coefficients.float() / coefficient_scale,
        n_points=max(128, int(n_directions) * 2))
    target_curve = sample_fourier_curve(
        target_center_norm, target_coefficients.float() / coefficient_scale,
        n_points=max(128, int(n_directions) * 2))

    count = max(int(n_directions), 8)
    angles = torch.arange(
        count, device=pred_center.device, dtype=pred_center.dtype
    ) * (2.0 * math.pi / count)
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    pred_offset = pred_curve - pred_center_norm[:, None, :]
    target_offset = target_curve - target_center_norm[:, None, :]
    pred_support = torch.einsum(
        'npc,dc->npd', pred_offset, directions).amax(dim=1)
    target_support = torch.einsum(
        'npc,dc->npd', target_offset, directions).amax(dim=1)

    under = F.relu(target_support - pred_support)
    over = F.relu(pred_support - target_support - float(radial_band))
    return (float(under_gain) * under.mean() +
            float(over_gain) * over.mean())


def fourier_point_coverage_loss(pred_center, pred_coefficients, target_center, matched_batch,
                                macro_targets, micro_targets, grid_size,
                                n_directions=32, margin_gain=1.0,
                                envelope_padding=0.0, support_tolerance=0.0,
                                overcoverage_gain=0.0, radial_band=0.0,
                                hard_fraction=0.0, hard_gain=0.0):
    """Use single-target GT to penalize Fourier contours that under-cover targets radially.

    Covered targets produce zero loss, so only missed targets effectively support the contour.
    Point-head predictions are deliberately not consumed here: the point head is an independent
    detached auxiliary task, while reliable single-target GT supervises macro contour coverage.
    Contours outside the point envelope are left to the Fourier GT losses.
    """
    zero = pred_center.sum() * 0.0
    if (pred_center.numel() == 0 or macro_targets.numel() == 0 or
            micro_targets.numel() == 0 or pred_coefficients.shape[-1] < 4):
        return zero

    height, width = int(grid_size[0]), int(grid_size[1])
    scale_xy = pred_center.new_tensor([width, height])
    pred_center_norm = pred_center.float() / scale_xy
    target_center_norm = target_center.float() / scale_xy

    terms = pred_coefficients.shape[-1] // 4
    coef_scale = pred_coefficients.new_tensor([width, width, height, height]).repeat(terms)
    pred_curve = sample_fourier_curve(
        pred_center_norm, pred_coefficients.float() / coef_scale,
        n_points=max(64, n_directions * 4))

    def radial_undercoverage(curve, points, boxes_wh, reference_center):
        """Compare each point edge with the curve radius in that point's own direction."""
        if points.numel() == 0:
            return None

        point_vectors = points - reference_center
        point_radii = torch.linalg.vector_norm(point_vectors, dim=1).clamp_min(1e-6)
        point_directions = point_vectors / point_radii[:, None]

        curve_vectors = curve - reference_center
        curve_radii = torch.linalg.vector_norm(curve_vectors, dim=1).clamp_min(1e-6)
        radial_projection = torch.einsum('pc,qc->pq', point_directions, curve_vectors)
        angular_similarity = radial_projection / curve_radii[None, :]


        neighbor_count = min(3, curve.shape[0])
        nearest = angular_similarity.topk(neighbor_count, dim=1).indices
        curve_reach = radial_projection.gather(1, nearest).amax(dim=1)

        box_margin = 0.5 * (
            boxes_wh[:, 0] * point_directions[:, 0].abs() +
            boxes_wh[:, 1] * point_directions[:, 1].abs()
        ) * float(margin_gain)
        required_reach = point_radii + box_margin + float(envelope_padding)
        undercoverage = F.relu(
            required_reach - curve_reach - float(support_tolerance))
        under_loss = undercoverage.mean()
        if float(hard_gain) > 0.0 and float(hard_fraction) > 0.0:


            hard_count = max(
                1, min(undercoverage.numel(),
                       int(math.ceil(undercoverage.numel() *
                                     float(hard_fraction)))))
            hard_loss = undercoverage.topk(hard_count).values.mean()
            under_loss = under_loss + float(hard_gain) * hard_loss
        if float(overcoverage_gain) <= 0.0:
            return under_loss


        point_angles = torch.atan2(point_vectors[:, 1], point_vectors[:, 0])
        sector = torch.floor(
            (point_angles + math.pi) / (2.0 * math.pi) * max(int(n_directions), 8)
        ).long().clamp(0, max(int(n_directions), 8) - 1)
        outer_indices = []
        for sector_id in torch.unique(sector):
            candidates = torch.where(sector == sector_id)[0]
            outer_indices.append(candidates[required_reach[candidates].argmax()])
        outer_indices = torch.stack(outer_indices)
        overcoverage = F.relu(
            curve_reach[outer_indices] - required_reach[outer_indices] -
            float(radial_band))
        return (under_loss +
                float(overcoverage_gain) * overcoverage.mean())

    macro_centers = macro_targets[:, 6:8].float()
    macro_batches = macro_targets[:, 0].long()
    micro_batches = micro_targets[:, 0].long()
    gt_losses = []

    for match_idx in range(pred_center.shape[0]):
        batch_idx = matched_batch[match_idx].long()
        macro_ids = torch.where(macro_batches == batch_idx)[0]
        micro_ids = torch.where(micro_batches == batch_idx)[0]
        if macro_ids.numel() == 0 or micro_ids.numel() == 0:
            continue


        matched_macro_local = torch.cdist(
            target_center_norm[match_idx:match_idx + 1], macro_centers[macro_ids]).argmin(dim=1)[0]
        micro_xy = micro_targets[micro_ids, 2:4].float()
        assigned_local = torch.cdist(micro_xy, macro_centers[macro_ids]).argmin(dim=1)
        selected = micro_ids[assigned_local == matched_macro_local]
        if selected.numel() == 0:
            continue

        reference_center = target_center_norm[match_idx].detach()
        gt_xy = micro_targets[selected, 2:4].float()
        gt_wh = micro_targets[selected, 4:6].float()


        gt_undercoverage = radial_undercoverage(
            pred_curve[match_idx], gt_xy, gt_wh, reference_center)
        if gt_undercoverage is not None:
            gt_losses.append(gt_undercoverage)

    loss_gt = torch.stack(gt_losses).mean() if gt_losses else zero
    return loss_gt


def select_unique_macro_matches(matched_batch, target_center, match_quality):
    """Select one representative anchor/grid match for each macro target on a feature level."""
    if matched_batch.numel() <= 1:
        return torch.arange(matched_batch.numel(), device=matched_batch.device)


    keys = torch.cat((matched_batch[:, None].to(target_center.dtype), target_center.detach()), dim=1)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    selected = []
    quality = match_quality.detach()
    for group_idx in range(unique_keys.shape[0]):
        candidates = torch.where(inverse == group_idx)[0]
        selected.append(candidates[quality[candidates].argmax()])
    return torch.stack(selected)

def point_hard_negative_peak_loss(
        pred_logits, point_target, topk=100, exclusion_radius=3,
        threshold=0.30):
    """Penalize inference-style local maxima outside GT protection disks.

    Peak/Top-K selection is deliberately detached. The returned BCE remains
    differentiable with respect to the selected ``pred_logits``.
    """
    batch = pred_logits.shape[0]
    probability = pred_logits.sigmoid()
    with torch.no_grad():
        detached_probability = probability.detach()
        local_max = F.max_pool2d(
            detached_probability, kernel_size=3, stride=1, padding=1)
        exclusion_radius = max(int(exclusion_radius), 0)
        if exclusion_radius > 0:
            kernel_size = 2 * exclusion_radius + 1
            protected = F.max_pool2d(
                point_target.detach(), kernel_size=kernel_size, stride=1,
                padding=exclusion_radius) > 0
        else:
            protected = point_target.detach() > 0
        peak_mask = (
            (detached_probability >= float(threshold)) &
            (detached_probability == local_max) &
            (~protected))

    losses = []
    selected_count = 0
    topk = max(int(topk), 1)
    for batch_index in range(batch):
        flat_mask = peak_mask[batch_index, 0].reshape(-1)
        candidate_count = int(flat_mask.sum().item())
        if candidate_count <= 0:
            continue
        select_count = min(topk, candidate_count)
        flat_score = detached_probability[
            batch_index, 0].reshape(-1).masked_fill(~flat_mask, -1.0)
        selected_index = flat_score.topk(
            select_count, largest=True).indices
        selected_logits = pred_logits[
            batch_index, 0].reshape(-1)[selected_index]
        losses.append(F.binary_cross_entropy_with_logits(
            selected_logits, torch.zeros_like(selected_logits),
            reduction='mean'))
        selected_count += select_count

    if not losses:
        return pred_logits.sum() * 0.0, pred_logits.new_zeros(1)
    loss = torch.stack(losses).mean()
    peaks_per_frame = pred_logits.new_tensor(
        float(selected_count) / float(max(batch, 1)))
    return loss, peaks_per_frame


class ComputeLoss:

    def __init__(self, model, autobalance=False,hungarian=False,sort_obj_iou=0,cm=None,ec=0.3):
        self.sort_obj_iou = sort_obj_iou
        device = next(model.parameters()).device
        model_yaml = model.module.yaml if is_parallel(model) else model.yaml
        self.pcbi_point_to_contour = bool(
            model_yaml.get('pcbi_point_to_contour', False) or
            model_yaml.get('pcbi_point_to_fourier', False))
        self.pcbi_pred_cover = bool(model_yaml.get(
            'pcbi_pred_cover', self.pcbi_point_to_contour))
        self.h = model.hyp
        self.GNF = self.h.get('GNF',0)
        self.pred_clockwise = self.h.get('pred_clockwise',0)
        self.hungarian_flag = hungarian

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['obj_pw']], device=device))
        self.order = {
            'Detect':None,
            'Detect2':None,
            'FTInfer':None,
            'SubclassInfer':None,
            'ValueInfer':None
        }

        start = 0
        for i, h in enumerate(model.yaml['head']):
            head_name = h[2]
            if head_name == 'FourierProposal':


                self.order['Detect'] = (start, start + 3)
                start += 3
            elif head_name in ['Detect', 'Detect2', 'FTInfer', 'SubclassInfer', 'ValueInfer']:
                self.order[head_name] = (start, start+3)
                start += 3


        if self.order['Detect2'] is not None:
            MSELoss = nn.MSELoss(reduction='mean')
            L1Loss = nn.L1Loss(reduction='mean')
            BCEdir = nn.SmoothL1Loss(reduction='mean')
            BCEab = nn.SmoothL1Loss(reduction='mean')
            BCEpab = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['pab_pw']], device=device))


        self.cp, self.cn = smooth_BCE(eps=self.h.get('label_smoothing', 0.0))


        g = self.h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        base_model = model.module if is_parallel(model) else model
        det = base_model.get_module_byname('Detect')
        proposal = base_model.get_module_byname('FourierProposal')
        det = det if det is not None else proposal
        if det is None:
            raise ValueError('ComputeLoss requires Detect or FourierProposal')
        self.fourier_proposal_mode = (
            proposal is not None and base_model.get_module_byname('Detect') is None)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(det.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, self.h, autobalance

        self.offset = self.h.get('offset', 1)
        if(self.offset==0):
            self.gr=0
        if self.fourier_proposal_mode:


            self.gr = 0


        self.bce_aux = nn.BCELoss()


        if self.order['Detect2'] is not None:
            self.MSELoss = MSELoss
            self.L1Loss = L1Loss
            self.BCEdir = BCEdir
            self.BCEab = BCEab
            self.BCEpab = BCEpab

        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))


        det2 = model.module.get_module_byname('Detect2') if is_parallel(model) else model.get_module_byname('Detect2')
        if det2 is not None:
            setattr(self, 'anchorsab', getattr(det2, 'anchors'))
        else:
            setattr(self, 'anchorsab', [None] * len(self.anchors))

        self.svinfer = 0
        det3 = model.module.get_module_byname('SubclassInfer') if is_parallel(model) else model.get_module_byname('SubclassInfer')
        if det3 is not None:
            setattr(self, 'ns', getattr(det3, 'ns'))
            self.BCEcls_sub = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h.get('cls_sub_pw',1)], device=device))
            self.svinfer += 1

        det4 = model.module.get_module_byname('ValueInfer') if is_parallel(model) else model.get_module_byname('ValueInfer')
        if det4 is not None:
            setattr(self, 'nv', getattr(det4, 'nv'))

            self.MSEvinfer = nn.SmoothL1Loss()
            self.svinfer += self.nv

        det_ft = model.module.get_module_byname('FTInfer') if is_parallel(model) else model.get_module_byname('FTInfer')
        if det_ft is not None:
            setattr(self, 'ft_coef', getattr(det_ft, 'ft_coef'))
            self.ft_len = 2 + 4 * self.ft_coef
            self.ftloss_xy = nn.SmoothL1Loss()
            self.ftloss_coef = nn.SmoothL1Loss()
            self.lamda = det_ft.lamda
            self.d0 = det_ft.d0
        else:
            self.ft_len = 0

        self.n_loop=model.hyp.get('n_loop',8)

        self.cm = cm.to(device=device)
        self.ec = ec

        self.linear_sum_assignment = linear_sum_assignment


        self.bce_p2p = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_p2p = nn.L1Loss(reduction='none')
        self.epoch = 0
        self.biocount_warmup_epochs = int(self.hyp.get('biocount_warmup_epochs', 5))

    def ftloss_coef_cm(self, ft_coef_pi, ft_coefs_labels):


        assert (ft_coef_pi.shape[-1] % 4)==0
        term = ft_coef_pi.shape[-1]//4
        assert term==self.cm.shape[0]


        total_loss = 0


        for i in range(term):

            start_idx = 4 * i
            end_idx = 4 * (i + 1)


            ft_coef_pi_segment = ft_coef_pi[:, start_idx:end_idx]
            ft_coefs_labels_segment = ft_coefs_labels[:, start_idx:end_idx]


            segment_loss = self.ftloss_coef(ft_coef_pi_segment, ft_coefs_labels_segment / (self.ec+self.cm[i]))


            total_loss += segment_loss

        return total_loss


    def __call__(self, p, targets, fold, mask_dir = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0], paths = None, guidance_map=None):
        device = targets.device
        targets_all_frames = targets

        p2p_preds = None
        p_yolo = p

        if isinstance(p[-1], dict) and ("pred_logits" in p[-1] or "pred_density" in p[-1]):
            p2p_preds = p[-1]
            p_yolo = p[:-1]


        bs_pred = p_yolo[0].shape[0] if not isinstance(p_yolo, tuple) else p_yolo[0][0].shape[0]


        point_batch = (int(p2p_preds['pred_logits'].shape[0])
                       if isinstance(p2p_preds, dict) and
                       isinstance(p2p_preds.get('pred_logits'), torch.Tensor)
                       else bs_pred)
        temporal_point_full_batch = (
            bs_pred > 0 and point_batch > bs_pred and point_batch % bs_pred == 0)
        seq_len = point_batch // bs_pred if temporal_point_full_batch else 1
        if seq_len == 1 and paths is not None and bs_pred > 0 and len(paths) > bs_pred and len(paths) % bs_pred == 0:
            seq_len = len(paths) // bs_pred
        elif paths is None and targets.shape[0] > 0 and targets[:, 0].max() >= bs_pred:

            seq_len = 5
        if seq_len > 1 and targets.shape[0] > 0:
            targets = targets[targets[:, 0].long() % seq_len == (seq_len - 1)].clone()
            targets[:, 0] = torch.div(targets[:, 0], seq_len, rounding_mode='floor')


        self.macro_targets_for_mask = targets[targets[:, 1] == 1.0].clone()


        targets_micro_point = (
            targets_all_frames[targets_all_frames[:, 1] == 0.0].clone()
            if temporal_point_full_batch else
            targets[targets[:, 1] == 0.0].clone())


        targets_micro_fourier = targets[targets[:, 1] == 0.0].clone()
        if targets_micro_fourier.numel() > 0:
            fourier_batch = targets_micro_fourier[:, 0].long()
            if fourier_batch.min() < 0 or fourier_batch.max() >= bs_pred:
                raise RuntimeError(
                    'Fourier micro-target batch indices are not aligned with '
                    f'last-frame predictions: range='
                    f'[{int(fourier_batch.min())}, {int(fourier_batch.max())}], '
                    f'prediction_batch={bs_pred}, seq_len={seq_len}')


        targets_yolo = self.macro_targets_for_mask.clone()
        if targets_yolo.shape[0] > 0:
            targets_yolo[:, 1] = 0.0


        guidance_map = None
        if isinstance(p_yolo, tuple):
            p_yolo, guidance_map = p_yolo

        dir_scale = torch.zeros(len(mask_dir), device=device)
        dir_scale[np.array(mask_dir) > 0] = 1.0

        pH = p_yolo[self.order['Detect'][0]:self.order['Detect'][1]]
        pD = p_yolo[self.order['Detect2'][0]:self.order['Detect2'][1]] if self.order['Detect2'] is not None else [None] * len(pH)
        pF = p_yolo[self.order['FTInfer'][0]:self.order['FTInfer'][1]] if self.order['FTInfer'] is not None else [None] * len(pH)
        pS = p_yolo[self.order['SubclassInfer'][0]:self.order['SubclassInfer'][1]] if self.order['SubclassInfer'] is not None else [None] * len(pH)
        pV = p_yolo[self.order['ValueInfer'][0]:self.order['ValueInfer'][1]] if self.order['ValueInfer'] is not None else [None] * len(pH)

        bs = pH[0].shape[0]
        assert (pF is None) or pF[0].shape[0]==bs

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        lf_xy = torch.zeros(1, device=device)
        lf_coef = torch.zeros(1, device=device)
        lf_curve = torch.zeros(1, device=device)
        lf_cover_gt = torch.zeros(1, device=device)
        lf_pred_cover = torch.zeros(1, device=device)
        lf_support_consistency = torch.zeros(1, device=device)
        support_store = GLOBAL_FORMATION_SUPPORT_DICT.get(device, None)

        tcls, tbox, indices, anchors, tdir, tab, anchorsab, svinfer, ftinfer, tiou = self.build_targets_whole(pH, pD, pF, pS, pV, targets_yolo, fold)


        for i, (pi, pi_dir, pi_f, pi_s, pi_v) in enumerate(zip(pH, pD, pF, pS, pV)):
            b, a, gj, gi, aab = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            nt = b.shape[0]
            if nt:

                dir_scale_objs = dir_scale[tcls[i]]
                assert(dir_scale_objs.shape[0]==nt)
                dir_set = dir_scale_objs > 0


                not_dir_set = torch.logical_not(dir_set)

                ps = pi[b, a, gj, gi]


                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze(-1)
                assert len(tiou)==0 or  iou.shape[0]==tiou[i].shape[0]

                if torch.nonzero(not_dir_set).shape[0] > 0:
                    lbox += (1.0 - iou[not_dir_set]).mean()
                if torch.nonzero(dir_set).shape[0] > 0:
                    lbox += self.L1Loss(pbox[dir_set,:2],tbox[i][dir_set,:2])


                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou < 2:
                    if self.sort_obj_iou==1:
                        sort_id = torch.argsort(score_iou)
                        b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou
                else:
                    assert self.sort_obj_iou==2
                    tobj[b, a, gj, gi] += (1.0 - self.gr) + self.gr * score_iou
                    tobj = torch.where(tobj > 1, torch.tensor(1), tobj)


                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(nt), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

                if pi_f is not None:
                    pi_f = pi_f[b, a, gj, gi]
                    assert nt==pi_f.shape[0]


                    ft_xy_pi = 0.5 + pi_f[:, :2]
                    lf_xy += self.ftloss_xy(ft_xy_pi, ftinfer[i][:, :2])

                    term = (pi_f.shape[1]-2) // 4
                    anchors_coef = anchors[i].view(nt, 2, 1).repeat(1, 1, 2).view(nt, -1)
                    ft_coef_pi = pi_f[:, 2:] * anchors_coef.repeat(1, term)

                    ft_coefs_labels = ftinfer[i][:, 2:]
                    if self.GNF > 0:
                        if self.pred_clockwise:
                            ft_correct_clockwise(ft_coef_pi, 0)
                        abcd1 = ft_coef_pi[:,:4].detach()
                        cos_sin_iou = solve_cos_sin_closed_form(ft_coefs_labels[:,:4],abcd1)
                        cos_sin_iou = angle_multiple(cos_sin_iou, term)
                        ft_coef_min = rotate_high_orders(None,ft_coefs_labels,cos_sin_iou,offset1=0)
                    else:
                        if self.n_loop==0:
                            ft_coef_min = ft_coefs_labels
                        else:
                            assert term==ft_coefs_labels.shape[1] // 4
                            ft_coef_min = loop_loss(self.n_loop,ft_coefs_labels,ft_coef_pi,self.cm,self.ec)

                    lf_coef += self.ftloss_coef_cm(ft_coef_pi, ft_coef_min)


                    ft_coef_decoded = (
                        ft_coef_pi.view(nt, term, 4) *
                        (self.ec + self.cm).to(ft_coef_pi).view(1, term, 1)
                    ).reshape(nt, -1)
                    pred_center_abs = torch.stack((ft_xy_pi[:, 0] + gi, ft_xy_pi[:, 1] + gj), dim=1)
                    target_center_abs = torch.stack(
                        (ftinfer[i][:, 0] + gi, ftinfer[i][:, 1] + gj), dim=1)
                    geometry_idx = select_unique_macro_matches(b, target_center_abs, iou)

                    support_consistency_weight = float(
                        self.hyp.get('ft_support_consistency', 0.0))
                    support_logits_level = get_support_level(
                        support_store, pi.shape[2:4], bs)
                    if (support_consistency_weight > 0.0 and
                            support_logits_level is not None and
                            geometry_idx.numel() > 0):
                        support_consistency = fourier_support_consistency_loss(
                            pred_center=pred_center_abs[geometry_idx],
                            pred_coefficients=ft_coef_decoded[geometry_idx],
                            target_center=target_center_abs[geometry_idx],
                            matched_batch=b[geometry_idx],
                            macro_targets=self.macro_targets_for_mask,
                            support_logits=support_logits_level,
                            grid_size=pi.shape[2:4],
                            n_directions=int(self.hyp.get('ft_support_directions', 32)),
                            support_threshold=float(self.hyp.get('ft_support_threshold', 0.10)),
                            undercoverage_gain=float(self.hyp.get('ft_support_under_gain', 2.0)))
                        warmup_epochs = int(self.hyp.get('ft_support_warmup_epochs', 10))
                        warmup = 1.0 if warmup_epochs <= 0 else min(
                            1.0, (float(getattr(self, 'epoch', 0)) + 1.0) / warmup_epochs)
                        lf_support_consistency += (
                            support_consistency * support_consistency_weight * warmup)


                    pred_cover_weight = float(self.hyp.get('ft_cover_pred', 0.0))
                    pred_point_logits = (
                        p2p_preds.get('pred_logits')
                        if isinstance(p2p_preds, dict) else None)
                    if (self.pcbi_pred_cover and pred_cover_weight > 0.0 and
                            pred_point_logits is not None and geometry_idx.numel() > 0):
                        pred_cover = fourier_support_consistency_loss(
                            pred_center=pred_center_abs[geometry_idx],
                            pred_coefficients=ft_coef_decoded[geometry_idx],
                            target_center=target_center_abs[geometry_idx],
                            matched_batch=b[geometry_idx],
                            macro_targets=self.macro_targets_for_mask,
                            support_logits=pred_point_logits,
                            grid_size=pi.shape[2:4],
                            n_directions=int(
                                self.hyp.get('ft_cover_pred_directions', 32)),
                            support_threshold=float(
                                self.hyp.get('ft_cover_pred_threshold', 0.10)),
                            undercoverage_gain=float(
                                self.hyp.get('ft_cover_pred_under_gain', 2.0)),
                            boundary_gain=float(
                                self.hyp.get(
                                    'ft_cover_pred_boundary_gain', 0.0)),
                            preserve_peaks=True)
                        pred_warmup_epochs = int(
                            self.hyp.get('ft_cover_pred_warmup_epochs', 20))
                        pred_warmup = 1.0 if pred_warmup_epochs <= 0 else min(
                            1.0, (float(getattr(self, 'epoch', 0)) + 1.0) /
                            pred_warmup_epochs)
                        weighted_pred_cover = (
                            pred_cover * pred_cover_weight * pred_warmup)
                        lf_pred_cover += weighted_pred_cover
                        lf_support_consistency += weighted_pred_cover

                    if float(self.hyp.get('ft_cover_gt', 0.0)) > 0.0:
                        cover_gt = fourier_point_coverage_loss(
                            pred_center=pred_center_abs[geometry_idx],
                            pred_coefficients=ft_coef_decoded[geometry_idx],
                            target_center=target_center_abs[geometry_idx],
                            matched_batch=b[geometry_idx],
                            macro_targets=self.macro_targets_for_mask,
                            micro_targets=targets_micro_fourier,
                            grid_size=pi.shape[2:4],
                            n_directions=int(self.hyp.get('ft_cover_directions', 32)),
                            margin_gain=float(self.hyp.get('ft_cover_margin', 1.0)),
                            envelope_padding=float(self.hyp.get('ft_cover_padding', 0.0)),
                            support_tolerance=float(self.hyp.get('ft_cover_tolerance', 0.0)),
                            overcoverage_gain=float(
                                self.hyp.get('ft_cover_over_gain', 0.0)),
                            radial_band=float(self.hyp.get('ft_cover_band', 0.0)),
                            hard_fraction=float(
                                self.hyp.get('ft_cover_hard_fraction', 0.0)),
                            hard_gain=float(
                                self.hyp.get('ft_cover_hard_gain', 0.0)))
                        lf_cover_gt += cover_gt


                    curve_weight = float(self.hyp.get('ft_curve', 0.0))
                    area_weight = float(self.hyp.get('ft_area', 0.0))
                    shape_weight = float(self.hyp.get('ft_shape', 0.0))
                    harmonic_shape_weight = float(
                        self.hyp.get('ft_harmonic_shape', 0.0))
                    directional_weight = float(
                        self.hyp.get('ft_directional', 0.0))
                    if (curve_weight > 0.0 or area_weight > 0.0 or
                            shape_weight > 0.0 or harmonic_shape_weight > 0.0 or
                            directional_weight > 0.0
                            ) and geometry_idx.numel() > 0:
                        curve_loss, area_loss = fourier_curve_geometry_loss(
                            pred_center_abs[geometry_idx],
                            ft_coef_decoded[geometry_idx],
                            target_center_abs[geometry_idx],
                            ft_coef_min[geometry_idx],
                            n_points=int(self.hyp.get('ft_curve_points', 96)))
                        warmup_epochs = int(self.hyp.get('ft_curve_warmup_epochs', 0))
                        warmup = 1.0 if warmup_epochs <= 0 else min(
                            1.0, (float(getattr(self, 'epoch', 0)) + 1.0) / warmup_epochs)


                        lf_curve += (
                            curve_loss * curve_weight + area_loss * area_weight
                        ) * warmup
                        if shape_weight > 0.0:
                            shape_loss = fourier_normalized_shape_loss(
                                pred_center_abs[geometry_idx],
                                ft_coef_decoded[geometry_idx],
                                target_center_abs[geometry_idx],
                                ft_coef_min[geometry_idx],
                                n_points=int(self.hyp.get('ft_curve_points', 96)))
                            lf_curve += shape_loss * shape_weight * warmup
                        if harmonic_shape_weight > 0.0:
                            harmonic_shape_loss = fourier_harmonic_shape_loss(
                                ft_coef_decoded[geometry_idx],
                                ft_coef_min[geometry_idx])
                            lf_curve += (
                                harmonic_shape_loss * harmonic_shape_weight * warmup)
                        if directional_weight > 0.0:
                            directional_loss = fourier_directional_support_loss(
                                pred_center_abs[geometry_idx],
                                ft_coef_decoded[geometry_idx],
                                target_center_abs[geometry_idx],
                                ft_coef_min[geometry_idx],
                                grid_size=pi.shape[2:4],
                                n_directions=int(
                                    self.hyp.get('ft_directional_directions', 64)),
                                under_gain=float(
                                    self.hyp.get('ft_directional_under', 2.0)),
                                over_gain=float(
                                    self.hyp.get('ft_directional_over', 0.5)),
                                radial_band=float(
                                    self.hyp.get('ft_directional_band', 0.00625)))
                            lf_curve += directional_loss * directional_weight * warmup

            pobj = pi[..., 4]
            if self.h.get('pn_obj', 1) == 1:
                obji = self.BCEobj(pobj, tobj)
            else:
                mask = torch.rand_like(tobj) < self.h.get('pn_obj', 1)
                mask_obj,mask_noobj = tobj > 0, (tobj == 0) & mask
                pim = torch.cat([pobj[mask_obj], pobj[mask_noobj]], dim=0)
                tm = torch.cat([tobj[mask_obj], tobj[mask_noobj]], dim=0)
                pn_scale = 1
                obji = pn_scale * self.BCEobj(pim, tm)

            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]

        lmask = torch.zeros(1, device=device)
        mask_weight = float(self.hyp.get('mask_loss', 0.5))

        guidance_map = GLOBAL_MASK_DICT.get(device, None)

        if guidance_map is not None and mask_weight > 0.0:
            B, _, H, W = guidance_map.shape

            micro_targets = targets_micro_point
            macro_targets = self.macro_targets_for_mask


            micro_mask = torch.zeros_like(guidance_map)
            if micro_targets.shape[0] > 0:
                b_idx = micro_targets[:, 0].long()
                grid_x = (micro_targets[:, 2] * W).long().clamp(0, W - 1)
                grid_y = (micro_targets[:, 3] * H).long().clamp(0, H - 1)
                micro_mask[b_idx, 0, grid_y, grid_x] = 1.0
                micro_mask = F.max_pool2d(micro_mask, kernel_size=3, stride=1, padding=1)


            macro_mask = torch.zeros_like(guidance_map)
            if macro_targets.shape[0] > 0:
                macro_mask_np = np.zeros((B, H, W), dtype=np.float32)

                for i in range(macro_targets.shape[0]):
                    b_idx = int(macro_targets[i, 0].item())

                    terms = (macro_targets.shape[1] - 8) // 4
                    a0 = macro_targets[i, 6].item() * W
                    c0 = macro_targets[i, 7].item() * H

                    npts = 128
                    t = np.linspace(0, 2 * np.pi, npts, endpoint=False)

                    if terms > 0:
                        coefs = macro_targets[i, 8:].detach().cpu().numpy()
                        a_k = coefs[0::4] * W
                        b_k = coefs[1::4] * W
                        c_k = coefs[2::4] * H
                        d_k = coefs[3::4] * H

                        k_array = np.arange(1, terms + 1)[:, None]
                        cos_t = np.cos(k_array * t)
                        sin_t = np.sin(k_array * t)

                        x_pts = a0 + np.dot(a_k, cos_t) + np.dot(b_k, sin_t)
                        y_pts = c0 + np.dot(c_k, cos_t) + np.dot(d_k, sin_t)
                    else:
                        w_box = macro_targets[i, 4].item() * W
                        h_box = macro_targets[i, 5].item() * H
                        x_pts = np.array([a0-w_box/2, a0+w_box/2, a0+w_box/2, a0-w_box/2])
                        y_pts = np.array([c0-h_box/2, c0-h_box/2, c0+h_box/2, c0+h_box/2])

                    pts = np.stack((x_pts, y_pts), axis=1).astype(np.int32)
                    cv2.fillPoly(macro_mask_np[b_idx], [pts], 1.0)

                macro_mask[:, 0, :, :] = torch.from_numpy(macro_mask_np).to(device)


            pos_mask = (micro_mask > 0).float()
            neg_inner_mask = ((micro_mask == 0) & (macro_mask > 0)).float()
            out_bounds_mask = (macro_mask == 0).float()

            num_pos = torch.clamp(pos_mask.sum(), min=1.0)
            num_neg_inner = torch.clamp(neg_inner_mask.sum(), min=1.0)
            num_out = torch.clamp(out_bounds_mask.sum(), min=1.0)

            loss_pos = torch.sum(pos_mask * (guidance_map - 1.0)**2) / num_pos
            loss_neg_inner = torch.sum(neg_inner_mask * (guidance_map - 0.0)**2) / num_neg_inner
            loss_out = torch.sum(out_bounds_mask * (guidance_map - 0.0)**2) / num_out

            lmask = loss_pos * 5.0 + loss_neg_inner * 1.0 + loss_out * 5.0

        GLOBAL_MASK_DICT[device] = None

        lp2p_cls = torch.zeros(1, device=device)
        lp2p_reg = torch.zeros(1, device=device)
        lp2p_wh = torch.zeros(1, device=device)
        lmotion_gate = torch.zeros(1, device=device)
        lhard_peak = torch.zeros(1, device=device)
        hard_peak_count = torch.zeros(1, device=device)
        lcandidate_hard_positive = torch.zeros(1, device=device)
        lguided_point = torch.zeros(1, device=device)
        candidate_hard_positive_count = torch.zeros(1, device=device)
        candidate_prior_mean = torch.zeros(1, device=device)
        candidate_prior_max = torch.zeros(1, device=device)
        candidate_recall_gain = torch.zeros(1, device=device)
        lp2p_total = torch.zeros(1, device=device)
        lformation_support = torch.zeros(1, device=device)

        if p2p_preds is not None and "pred_logits" in p2p_preds:
            pred_logits = p2p_preds["pred_logits"]
            pred_offsets = p2p_preds.get("pred_offsets", None)
            pred_wh = p2p_preds.get("pred_wh", None)

            B, _, H_map, W_map = pred_logits.shape

            point_target = torch.zeros_like(pred_logits)
            heat_target = torch.zeros_like(pred_logits)
            offset_target = torch.zeros(B, 2, H_map, W_map, device=device, dtype=pred_logits.dtype)
            offset_mask = torch.zeros_like(pred_logits)
            wh_target = torch.zeros(B, 2, H_map, W_map, device=device, dtype=pred_logits.dtype)
            if targets_micro_point.shape[0] > 0:
                b_idx = targets_micro_point[:, 0].long().clamp(0, B - 1)
                gx = targets_micro_point[:, 2] * W_map
                gy = targets_micro_point[:, 3] * H_map
                grid_x = gx.long().clamp(0, W_map - 1)
                grid_y = gy.long().clamp(0, H_map - 1)
                point_target[b_idx, 0, grid_y, grid_x] = 1.0
                heat_target[b_idx, 0, grid_y, grid_x] = 1.0
                dx = (gx.to(dtype=offset_target.dtype) - grid_x.to(dtype=offset_target.dtype)).clamp(0.0, 1.0)
                dy = (gy.to(dtype=offset_target.dtype) - grid_y.to(dtype=offset_target.dtype)).clamp(0.0, 1.0)
                offset_target[b_idx, 0, grid_y, grid_x] = dx
                offset_target[b_idx, 1, grid_y, grid_x] = dy
                offset_mask[b_idx, 0, grid_y, grid_x] = 1.0
                wh_target[b_idx, 0, grid_y, grid_x] = targets_micro_point[:, 4].to(pred_logits.dtype) * W_map
                wh_target[b_idx, 1, grid_y, grid_x] = targets_micro_point[:, 5].to(pred_logits.dtype) * H_map

            sigma = float(self.hyp.get('biopoint_sigma', self.hyp.get('biocount_sigma', 1.0)))
            radius = max(1, int(round(3 * sigma)))
            coords = torch.arange(-radius, radius + 1, device=device, dtype=pred_logits.dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing='ij')
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            kernel = kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)
            heat_target = F.conv2d(heat_target, kernel, padding=radius).clamp_(0.0, 1.0)

            use_macro_gt = bool(self.hyp.get('biopoint_use_macro_gt', True))
            macro_mask = torch.zeros_like(pred_logits)
            if use_macro_gt and self.macro_targets_for_mask.shape[0] > 0:
                macro_mask_np = np.zeros((B, H_map, W_map), dtype=np.float32)
                macro_targets = self.macro_targets_for_mask
                for mi in range(macro_targets.shape[0]):
                    b_now = int(macro_targets[mi, 0].item())
                    if b_now < 0 or b_now >= B:
                        continue
                    terms = (macro_targets.shape[1] - 8) // 4
                    a0 = macro_targets[mi, 6].item() * W_map
                    c0 = macro_targets[mi, 7].item() * H_map
                    if terms > 0:
                        npts = 128
                        t = np.linspace(0, 2 * np.pi, npts, endpoint=False)
                        coefs = macro_targets[mi, 8:].detach().cpu().numpy()
                        a_k = coefs[0::4] * W_map
                        b_k = coefs[1::4] * W_map
                        c_k = coefs[2::4] * H_map
                        d_k = coefs[3::4] * H_map
                        k_array = np.arange(1, terms + 1)[:, None]
                        x_pts = a0 + np.dot(a_k, np.cos(k_array * t)) + np.dot(b_k, np.sin(k_array * t))
                        y_pts = c0 + np.dot(c_k, np.cos(k_array * t)) + np.dot(d_k, np.sin(k_array * t))
                    else:
                        w_box = macro_targets[mi, 4].item() * W_map
                        h_box = macro_targets[mi, 5].item() * H_map
                        x_pts = np.array([a0 - w_box / 2, a0 + w_box / 2, a0 + w_box / 2, a0 - w_box / 2])
                        y_pts = np.array([c0 - h_box / 2, c0 - h_box / 2, c0 + h_box / 2, c0 + h_box / 2])
                    pts = np.stack((x_pts, y_pts), axis=1).astype(np.int32)
                    cv2.fillPoly(macro_mask_np[b_now], [pts], 1.0)
                macro_mask[:, 0] = torch.from_numpy(macro_mask_np).to(device)
            else:
                macro_mask.fill_(1.0)

            heat_target = heat_target * macro_mask
            outside_mask = 1.0 - macro_mask
            pred_prob = pred_logits.sigmoid()


            candidate_positive_weight = float(self.hyp.get(
                'biopoint_candidate_hard_positive', 0.0))
            candidate_prior = p2p_preds.get('candidate_recall_prior')
            configured_candidate_gain = p2p_preds.get('candidate_recall_gain')
            if torch.is_tensor(configured_candidate_gain):
                candidate_recall_gain = configured_candidate_gain.detach().to(
                    device=device, dtype=pred_logits.dtype).mean().view(1)
            if (candidate_positive_weight > 0.0 and
                    torch.is_tensor(candidate_prior)):
                candidate_prior = candidate_prior.detach().to(
                    device=device, dtype=pred_logits.dtype)
                if candidate_prior.shape[-2:] != (H_map, W_map):
                    candidate_prior = F.interpolate(
                        candidate_prior, size=(H_map, W_map),
                        mode='bilinear', align_corners=False)
                if candidate_prior.shape[0] != B:
                    if candidate_prior.shape[0] % B == 0:
                        sequence_length = candidate_prior.shape[0] // B
                        candidate_prior = candidate_prior[
                            sequence_length - 1::sequence_length]
                    else:
                        candidate_prior = None
                if candidate_prior is not None:
                    candidate_prior_mean = candidate_prior.mean().detach().view(1)
                    candidate_prior_max = candidate_prior.amax().detach().view(1)
                    hard_positive_gamma = max(float(self.hyp.get(
                        'biopoint_candidate_hard_positive_gamma', 2.0)), 0.0)
                    difficulty = (1.0 - pred_prob.detach()).pow(
                        hard_positive_gamma)
                    hard_positive_weight = (
                        point_target * candidate_prior.clamp(0.0, 1.0) *
                        difficulty)
                    hard_positive_norm = hard_positive_weight.sum()
                    if hard_positive_norm > 0:
                        lcandidate_hard_positive = (
                            F.softplus(-pred_logits) *
                            hard_positive_weight).sum() / hard_positive_norm
                    candidate_hard_positive_count = (
                        (hard_positive_weight > 0).float().sum() /
                        max(float(B), 1.0))

            focal_gamma = float(self.hyp.get('biopoint_focal_gamma', 2.0))
            bce_map = F.binary_cross_entropy_with_logits(pred_logits, heat_target, reduction='none')
            focal_weight = (pred_prob - heat_target).abs().pow(focal_gamma)
            pos_weight = 1.0 + heat_target * float(self.hyp.get('biopoint_pos_gain', 12.0))
            valid_weight = macro_mask + outside_mask * float(self.hyp.get('biopoint_outside_weight', 0.25))
            lp2p_cls = (bce_map * focal_weight * pos_weight * valid_weight).sum() / torch.clamp(valid_weight.sum(), min=1.0)


            guided_point_weight = float(
                self.hyp.get('biopoint_guided_loss', 0.0))
            guided_logits = p2p_preds.get('guided_pred_logits')
            fourier_point_gate = p2p_preds.get('fourier_point_gate')
            if (guided_point_weight > 0.0 and
                    torch.is_tensor(guided_logits)):
                if guided_logits.shape[-2:] != (H_map, W_map):
                    guided_logits = F.interpolate(
                        guided_logits, size=(H_map, W_map),
                        mode='bilinear', align_corners=False)


                if (torch.is_tensor(fourier_point_gate) and
                        fourier_point_gate.shape[0] == B):
                    gate = fourier_point_gate.detach().to(
                        device=device, dtype=torch.float32)
                    if gate.shape[-2:] != (H_map, W_map):
                        gate = F.interpolate(
                            gate, size=(H_map, W_map), mode='bilinear',
                            align_corners=False)
                    protect_threshold = min(max(float(self.hyp.get(
                        'biopoint_guided_gt_protect_threshold', 0.05)),
                        0.0), 1.0)
                    gt_protect = (heat_target.float() >= protect_threshold)
                    protected_gate = torch.where(
                        gt_protect, torch.ones_like(gate), gate)
                    raw_logits_for_guidance = p2p_preds.get(
                        'raw_pred_logits', pred_logits)
                    if raw_logits_for_guidance.shape[-2:] != (H_map, W_map):
                        raw_logits_for_guidance = F.interpolate(
                            raw_logits_for_guidance, size=(H_map, W_map),
                            mode='bilinear', align_corners=False)
                    raw_logits_f32 = raw_logits_for_guidance.float()
                    log_probability = (
                        F.logsigmoid(raw_logits_f32) +
                        torch.log(protected_gate.clamp_min(1e-4)))
                    probability = torch.exp(log_probability)
                    guided_logits = (
                        log_probability - torch.log1p(
                            -probability.clamp(
                                max=1.0 - torch.finfo(torch.float32).eps))
                    ).to(pred_logits.dtype)
                guided_probability = guided_logits.sigmoid()
                guided_bce = F.binary_cross_entropy_with_logits(
                    guided_logits, heat_target, reduction='none')
                guided_focal = (
                    guided_probability - heat_target).abs().pow(focal_gamma)
                lguided_point = (
                    guided_bce * guided_focal * pos_weight * valid_weight
                ).sum() / torch.clamp(valid_weight.sum(), min=1.0)

            center_bce = F.binary_cross_entropy_with_logits(pred_logits, point_target, reduction='none')
            center_pos = point_target
            center_neg = (1.0 - point_target) * macro_mask
            center_pos_gain = float(self.hyp.get('biopoint_center_pos_gain', 80.0))
            center_neg_gain = float(self.hyp.get('biopoint_center_neg_gain', 1.0))
            center_weight = (
                center_pos * center_pos_gain + center_neg * center_neg_gain)
            center_norm = torch.clamp(
                center_pos.sum() * center_pos_gain +
                center_neg.sum() * center_neg_gain, min=1.0)
            lp2p_center = (center_bce * center_weight).sum() / center_norm

            if pred_offsets is not None and offset_mask.sum() > 0:
                lp2p_reg = (F.smooth_l1_loss(pred_offsets, offset_target, reduction='none') *
                            offset_mask).sum() / torch.clamp(offset_mask.sum() * 2.0, min=1.0)
            if pred_wh is not None and offset_mask.sum() > 0:
                lp2p_wh = (F.smooth_l1_loss(pred_wh, wh_target, reduction='none') *
                           offset_mask).sum() / torch.clamp(offset_mask.sum() * 2.0, min=1.0)

            outside_loss = (pred_prob * outside_mask).sum() / torch.clamp(outside_mask.sum(), min=1.0)


            hard_peak_weight = float(
                self.hyp.get('biopoint_hard_peak', 0.0))
            if hard_peak_weight > 0.0:
                lhard_peak, hard_peak_count = (
                    point_hard_negative_peak_loss(
                        pred_logits, point_target,
                        topk=self.hyp.get(
                            'biopoint_hard_peak_topk', 100),
                        exclusion_radius=self.hyp.get(
                            'biopoint_hard_peak_exclusion_radius', 3),
                        threshold=min(max(float(self.hyp.get(
                            'biopoint_hard_peak_threshold', 0.30)),
                            0.0), 1.0)))

            motion_entry = GLOBAL_POINT_MOTION_DICT.get(device)
            motion_gate_weight = float(
                self.hyp.get('biopoint_motion_gate', 0.0))
            if (motion_gate_weight > 0.0 and
                    isinstance(motion_entry, dict)):
                motion_logits = motion_entry.get('gate_logits')
                motion_seq_len = max(
                    int(motion_entry.get('seq_len', 1)), 1)
                if torch.is_tensor(motion_logits):
                    if motion_logits.shape[0] != B:
                        if (motion_logits.shape[0] % motion_seq_len == 0 and
                                B == motion_logits.shape[0] //
                                motion_seq_len):
                            motion_logits = motion_logits[
                                motion_seq_len - 1::motion_seq_len]
                        else:
                            motion_logits = None
                    if motion_logits is not None:
                        if motion_logits.shape[-2:] != (H_map, W_map):
                            motion_logits = F.interpolate(
                                motion_logits, size=(H_map, W_map),
                                mode='bilinear', align_corners=False)
                        motion_target = heat_target.detach()
                        motion_prob = motion_logits.sigmoid()
                        motion_bce = F.binary_cross_entropy_with_logits(
                            motion_logits, motion_target,
                            reduction='none')
                        motion_gamma = float(self.hyp.get(
                            'biopoint_motion_focal_gamma', 2.0))
                        motion_focal = (
                            motion_prob - motion_target).abs().pow(
                                motion_gamma)
                        motion_pos_gain = float(self.hyp.get(
                            'biopoint_motion_pos_gain', 40.0))
                        motion_pixel_weight = (
                            1.0 + motion_target * motion_pos_gain)

                        frame_weight = torch.ones(
                            B, device=device, dtype=motion_logits.dtype)
                        configured_weights = self.hyp.get(
                            'biopoint_motion_frame_weights',
                            [0.0, 0.25, 0.50, 0.75, 1.0])
                        if (isinstance(configured_weights, (list, tuple)) and
                                len(configured_weights) == motion_seq_len and
                                B % motion_seq_len == 0):
                            sequence_weight = motion_logits.new_tensor(
                                configured_weights)
                            frame_weight = sequence_weight.repeat(
                                B // motion_seq_len)
                        frame_weight = frame_weight.view(B, 1, 1, 1)
                        weighted_motion = (
                            motion_bce * motion_focal *
                            motion_pixel_weight * frame_weight)
                        motion_norm = (
                            motion_pixel_weight * frame_weight).sum().clamp_min(
                                1.0)
                        lmotion_gate = (
                            weighted_motion.sum() / motion_norm)

            warmup = 1.0
            if self.biocount_warmup_epochs > 0:
                warmup = min(1.0, max(0.0, (float(getattr(self, 'epoch', 0)) + 1.0) / float(self.biocount_warmup_epochs)))
            hard_peak_warmup_epochs = max(int(
                self.hyp.get(
                    'biopoint_hard_peak_warmup_epochs', 0)), 0)
            hard_peak_warmup = (
                1.0 if hard_peak_warmup_epochs <= 0 else min(
                    1.0,
                    (float(getattr(self, 'epoch', 0)) + 1.0) /
                    float(hard_peak_warmup_epochs)))
            lp2p_total = (
                lp2p_cls * self.hyp.get('biopoint_heatmap', self.hyp.get('biocount_peak', 0.20)) +
                lp2p_center * self.hyp.get('biopoint_center', 0.30) +
                lp2p_reg * self.hyp.get('biopoint_offset', 0.05) +
                lp2p_wh * self.hyp.get('biopoint_wh', 0.05) +
                outside_loss * self.hyp.get('biopoint_region', self.hyp.get('biocount_region', 0.02)) +
                lmotion_gate * motion_gate_weight +
                lhard_peak * hard_peak_weight * hard_peak_warmup +
                lcandidate_hard_positive * candidate_positive_weight +
                lguided_point * guided_point_weight
            ) * warmup
        motion_entry = GLOBAL_POINT_MOTION_DICT.get(device)
        self.last_point_motion = {
            'loss': lmotion_gate.detach(),
            'hard_peak_loss': lhard_peak.detach(),
            'hard_peak_count': hard_peak_count.detach(),
            'candidate_hard_positive_loss': lcandidate_hard_positive.detach(),
            'candidate_hard_positive_count': candidate_hard_positive_count.detach(),
            'candidate_prior_mean': candidate_prior_mean.detach(),
            'candidate_prior_max': candidate_prior_max.detach(),
            'candidate_recall_gain': candidate_recall_gain.detach(),
            'guided_point_loss': lguided_point.detach(),
            'gate_mean': (
                motion_entry.get('gate').detach().mean()
                if isinstance(motion_entry, dict) and
                torch.is_tensor(motion_entry.get('gate'))
                else torch.zeros(1, device=device)),
            'gate_max': (
                motion_entry.get('gate').detach().amax()
                if isinstance(motion_entry, dict) and
                torch.is_tensor(motion_entry.get('gate'))
                else torch.zeros(1, device=device)),
        }
        GLOBAL_POINT_MOTION_DICT[device] = None


        support_weight = float(self.hyp.get('formation_support', 0.0))
        support_losses = []
        if isinstance(support_store, dict) and support_weight > 0.0:
            for entry in support_store.values():
                if not isinstance(entry, dict):
                    continue
                support_logits = select_last_sequence_support(entry.get('logits'), bs)
                if support_logits is None:
                    continue
                _, _, support_h, support_w = support_logits.shape
                support_target = rasterize_fourier_macro_mask(
                    self.macro_targets_for_mask, bs, support_h, support_w,
                    device=device, dtype=support_logits.dtype)
                pos_gain = float(self.hyp.get('formation_support_pos_gain', 3.0))
                bce = F.binary_cross_entropy_with_logits(
                    support_logits, support_target, reduction='none')
                pixel_weight = 1.0 + support_target * pos_gain
                support_bce = (
                    (bce * pixel_weight).sum() /
                    pixel_weight.sum().clamp_min(1.0))
                support_prob = support_logits.sigmoid()
                intersection = (support_prob * support_target).sum(dim=(1, 2, 3))
                union = (
                    support_prob.sum(dim=(1, 2, 3)) +
                    support_target.sum(dim=(1, 2, 3)))
                support_dice = (
                    1.0 - (2.0 * intersection + 1.0) / (union + 1.0)).mean()
                dice_weight = float(self.hyp.get('formation_support_dice', 0.5))
                support_losses.append(support_bce + dice_weight * support_dice)
        if support_losses:
            lformation_support = torch.stack(support_losses).mean() * support_weight
        GLOBAL_FORMATION_SUPPORT_DICT[device] = None

        if pF[0] is not None:
            lf_xy *= self.hyp.get('ft_infer_xy', 0.25)
            lf_coef *= self.hyp.get('ft_infer_coef', 0.01)
            lf_cover_gt *= self.hyp.get('ft_cover_gt', 0.0)


        self.last_fourier_alignment = {
            'seq_len': int(seq_len),
            'point_micro_count': int(targets_micro_point.shape[0]),
            'fourier_micro_count': int(targets_micro_fourier.shape[0]),
            'cover_loss': lf_cover_gt.detach(),
            'pred_cover_loss': lf_pred_cover.detach(),
            'directional_loss': lf_curve.detach(),
        }

        total_loss = (
            lbox + lobj + lcls + lf_xy + lf_coef + lf_curve + lf_cover_gt +
            mask_weight * lmask + lp2p_total + lformation_support +
            lf_support_consistency) * bs


        loss_items = torch.cat((
            lbox, lobj, lf_xy, lf_coef, lp2p_total.view(1)
        )).detach()

        return total_loss, loss_items


    def build_targets_whole(self, pH, pD, pF, pS, pV, target_all, fold):

        na, nt, svlength = self.na, target_all.shape[0], self.svinfer
        tcls, tbox, indices, anch = [], [], [], []
        tdir, tab, anch2, tiou = [], [], [], []
        tft = []
        svinfer = []
        gain = torch.ones(7, device=target_all.device)
        gain_dir = torch.ones(4, device=target_all.device)
        gain_ft = torch.ones(self.ft_len, device=target_all.device)
        ai = torch.arange(na, device=target_all.device).float().view(na, 1).repeat(1, nt)
        has_D = pD[0] is not None
        has_F = pF[0] is not None
        has_S = pS[0] is not None
        has_V = pV[0] is not None

        targets = target_all[:, :6]
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        g = 0.5
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],

            ],
            device=target_all.device).float() * g
        if has_D:
            pts_targets = torch.clone(target_all[:, 6:14])
            dir_targets = pts2dir(pts_targets, fold_angle=fold)
            dir_targets = dir_targets.repeat(na, 1, 1)
        if has_F:
            start_ft = 14 if has_D else 6
            ft_targets = torch.clone(target_all[:, start_ft:start_ft + self.ft_len])
            ft_targets = ft_targets.repeat(na, 1, 1)
        for i in range(self.nl):
            anchors, shape = self.anchors[i], pH[i].shape
            anchors2 = self.anchorsab[i]
            t_dir = torch.zeros([0, 4], device=target_all.device)
            cms_infer = torch.zeros([0, svlength], device=target_all.device)
            t_ft = torch.zeros([0, self.ft_len], device=target_all.device)

            if nt:

                gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
                t = targets * gain


                r = t[:, :, 4:6] / anchors[:, None]


                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']


                t = t[j]
                if has_D:
                    gain_dir[2:4] = torch.tensor(pD[i].shape)[[3, 2]]

                    t_dir = dir_targets * gain_dir
                    t_dir = t_dir[j]

                if has_S or has_V:
                    cms_infer = torch.clone(target_all[:, 14 if has_D else 6:])
                    cms_infer = cms_infer.repeat(na, 1, 1)
                    cms_infer = cms_infer[j]

                if has_F:
                    gain_ft[0:2] = torch.tensor(pF[i].shape)[[3, 2]]

                    gain_ft[2:] = torch.tensor(pF[i].shape)[[3,3,2,2]].repeat((self.ft_len - 2) // 4)
                    t_ft = ft_targets * gain_ft
                    t_ft = t_ft[j]


                gxy = t[:, 2:4]
                if self.offset:
                    gxi = gain[[2, 3]] - gxy
                    j, k = ((gxy % 1 < g) & (gxy > 1)).T
                    l, m = ((gxi % 1 < g) & (gxi > 1)).T
                    j = torch.stack((torch.ones_like(j), j, k, l, m))
                    t = t.repeat((5, 1, 1))[j]
                    if has_D:
                        t_dir = t_dir.repeat((5, 1, 1))[j]
                    if has_S or has_V:
                        cms_infer = cms_infer.repeat((5, 1, 1))[j]
                    if has_F:
                        t_ft = t_ft.repeat((5, 1, 1))[j]
                    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                else:
                    offsets = torch.zeros_like(gxy)
                if anchors2 is not None and t_dir.shape[0] != 0:
                    max_an_iou_idx = torch.max(wh_iou(anchors2, t_dir[:, 2:4]), dim=0)[1]
                else:
                    max_an_iou_idx = torch.zeros(0, dtype=torch.long, device=t_dir.device)
            else:
                t = targets[0]
                offsets = 0
                max_an_iou_idx = torch.zeros(0, dtype=torch.long, device=t_dir.device)

            if self.hungarian_flag:


                a = t[:, 6].long()
                grid_box = t[:, 2:6].clone()
                grid_box[:, 0:2] = t[:, 2:4].floor() + 0.5
                grid_box[:, 2:] = anchors[a]
                gt_box = (target_all[:, :6].clone() * gain[:6])[:, 2:6]
                ious, mathcied_ious, (gt_idx, grid_idx) = hungarian_match(grid_box, gt_box)
                tiou.append(mathcied_ious)

                t_dir = t_dir[grid_idx]
                max_an_iou_idx = max_an_iou_idx[grid_idx]
                offsets = offsets[grid_idx]
                t = t[grid_idx]


            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T


            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1), max_an_iou_idx))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            if has_F:
                tft.append(torch.cat([t_ft[:, :2] - gij, t_ft[:, 2:]], 1))
            anch.append(anchors[a])
            tcls.append(c)

            tdir.append(t_dir[:, :2])
            tab.append(t_dir[:, 2:4])
            anch2.append(anchors2[max_an_iou_idx] if anchors2 is not None else [])
            svinfer.append(cms_infer)

        return tcls, tbox, indices, anch, tdir, tab, anch2, svinfer, tft, tiou

