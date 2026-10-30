"""Visualize Fourier coefficients emitted directly by FTInfer.

This script deliberately bypasses distance clustering, convex hull construction,
curve smoothing, and Fourier coefficient recomputation from individual detections.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from detect import detect_ft
from models.common import (
    DetectMultiBackend,
    GLOBAL_CANDIDATE_GEOMETRY_DICT,
    GLOBAL_ORACLE_PERIPHERAL_DICT,
    GLOBAL_POINT_FEATURE_DICT,
)
from tools.fft.fftinside import render_fourier_curve
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.ft_utils import fft_1, scale_coords_ft
from utils.general import check_file, check_img_size, increment_path, scale_coords
from utils.torch_utils import select_device


def _name_for_frame(path, dataset):
    path = Path(path)
    if getattr(dataset, "mode", "image") == "image":
        return path.stem
    return f"{path.stem}_{int(getattr(dataset, 'frame', 0)):06d}"


def _natural_path_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def _recursive_sequence_dirs(source_root):
    """Return leaf directories containing images, in deterministic order."""
    source_root = Path(source_root).resolve()
    parents = {
        path.parent
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
    }
    if not parents:
        raise FileNotFoundError(
            f"No image sequences found recursively below: {source_root}")
    return sorted(parents, key=lambda path: _natural_path_key(
        path.relative_to(source_root)))


def _iter_sequence_datasets(sequence_datasets):
    """Yield one stream while exposing sequence boundaries and active loader."""
    total = len(sequence_datasets)
    for sequence_index, (sequence_dir, dataset) in enumerate(sequence_datasets, 1):
        first = True
        for item in dataset:
            yield first, sequence_index, total, sequence_dir, dataset, item
            first = False


def _draw_raw_fourier(image, ft_pixels, color=(255, 255, 0), thickness=2, points=256):
    """Draw one pixel-coordinate FTInfer vector without changing its coefficients."""
    terms = (len(ft_pixels) - 2) // 4
    if terms <= 0:
        return None, None

    height, width = image.shape[:2]
    norm = np.asarray(ft_pixels, dtype=np.float32).copy()
    scale = np.asarray([width, height] + [width, width, height, height] * terms, dtype=np.float32)
    norm /= np.maximum(scale, 1.0)
    x, y = fft_1(norm.tolist(), points)
    polygon = np.stack((np.asarray(x) * width, np.asarray(y) * height), axis=1)
    polygon = np.round(polygon).astype(np.int32)
    cv2.polylines(image, [polygon], True, color, thickness, cv2.LINE_AA)
    return norm, polygon


def _read_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line in file:
            try:
                row = [float(value) for value in line.strip().split()]
            except ValueError:
                continue
            if row:
                rows.append(row)
    return rows


def _fourier_gt_polygon(row, width, height, points=256):
    """Convert one normalized ``class + Fourier coefficients`` GT row to pixels."""
    coefficients = np.asarray(row[1:], dtype=np.float32)
    terms = (len(coefficients) - 2) // 4
    if terms <= 0:
        return None
    expected = 2 + 4 * terms
    x, y = fft_1(coefficients[:expected].tolist(), points)
    polygon = np.stack((np.asarray(x) * width, np.asarray(y) * height), axis=1)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return np.round(polygon).astype(np.int32)


def _yolo_single(row, width, height):
    if len(row) < 5:
        return None
    xc, yc, bw, bh = row[1:5]
    center = (int(round(xc * width)), int(round(yc * height)))
    x1 = int(round((xc - bw / 2) * width))
    y1 = int(round((yc - bh / 2) * height))
    x2 = int(round((xc + bw / 2) * width))
    y2 = int(round((yc + bh / 2) * height))
    return center, (x1, y1, x2, y2)


def _polygon_iou(first, second, shape):
    if first is None or second is None or len(first) < 3 or len(second) < 3:
        return 0.0
    height, width = shape[:2]
    mask_first = np.zeros((height, width), dtype=np.uint8)
    mask_second = np.zeros_like(mask_first)
    cv2.fillPoly(mask_first, [first.reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mask_second, [second.reshape(-1, 1, 2)], 1)
    intersection = np.count_nonzero(mask_first & mask_second)
    union = np.count_nonzero(mask_first | mask_second)
    return float(intersection / union) if union else 0.0


def _greedy_polygon_matches(predictions, targets, shape):
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, target in enumerate(targets):
            candidates.append((_polygon_iou(prediction, target, shape), pred_index, gt_index))
    matches = {}
    used_targets = set()
    for iou, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in matches or gt_index in used_targets:
            continue
        matches[pred_index] = (gt_index, iou)
        used_targets.add(gt_index)
    return matches


def _point_inside(polygon, point):
    return (polygon is not None and len(polygon) >= 3 and
            cv2.pointPolygonTest(polygon.reshape(-1, 1, 2),
                                 (float(point[0]), float(point[1])), False) >= 0)


def _draw_text(image, text, origin, color, scale=0.42):
    x = max(2, min(int(origin[0]), image.shape[1] - 2))
    y = max(15, min(int(origin[1]), image.shape[0] - 3))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _paper_font_scale(image_width, point_size, display_width_inches=5.55,
                      thickness=1):
    """Map a typographic point size to OpenCV text at the paper display width."""
    pixels_per_inch = max(float(image_width) / float(display_width_inches), 1.0)
    target_height = float(point_size) * pixels_per_inch / 72.0
    base_height = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness)[0][1]
    return max(target_height / max(float(base_height), 1.0), 0.1)


def _draw_paper_text(image, text, origin, color, point_size=16,
                     display_width_inches=5.55):
    """Draw outlined text whose printed height follows the requested point size."""
    foreground_thickness = max(1, int(round(image.shape[1] / 700.0)))
    outline_thickness = foreground_thickness + max(
        2, int(round(image.shape[1] / 450.0)))
    scale = _paper_font_scale(
        image.shape[1], point_size, display_width_inches,
        foreground_thickness)
    text_width, text_height = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale,
        foreground_thickness)[0]
    x = max(5, min(int(origin[0]), image.shape[1] - text_width - 5))
    y = max(text_height + 5, min(int(origin[1]), image.shape[0] - 5))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, foreground_thickness, cv2.LINE_AA)


def _debug_tensor_for_frame(value, batch_index):
    """Return one diagnostic tensor as a CPU float32 NumPy array."""
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().float().cpu()
    if tensor.ndim >= 3:
        index = min(max(int(batch_index), 0), tensor.shape[0] - 1)
        tensor = tensor[index]
    return tensor.numpy()


def _debug_spatial_map(value, batch_index):
    """Reduce channels/anchors while preserving the final HxW map."""
    array = _debug_tensor_for_frame(value, batch_index)
    if array is None:
        return None
    while array.ndim > 2:
        array = np.nanmax(array, axis=0)
    if array.ndim != 2:
        return None
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _heatmap_panel(spatial_map, image, title):
    height, width = image.shape[:2]
    if spatial_map is None:
        spatial_map = np.zeros((height, width), dtype=np.float32)
    spatial_map = cv2.resize(
        spatial_map.astype(np.float32), (width, height),
        interpolation=cv2.INTER_LINEAR)
    minimum, maximum = float(spatial_map.min()), float(spatial_map.max())
    normalized = (
        (spatial_map - minimum) / max(maximum - minimum, 1e-8))
    colored = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.48, colored, 0.52, 0.0)
    header = 28
    panel = cv2.copyMakeBorder(
        overlay, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(
        panel, f"{title} [{minimum:.4g}, {maximum:.4g}]",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _point_temporal_adapter(core_model):
    """Return the explicit point-motion adapter from a loaded checkpoint."""
    point_head = (
        core_model.get_module_byname("BioCountInfer")
        if hasattr(core_model, "get_module_byname") else None)
    adapter = getattr(point_head, "temporal_adapter", None)
    if (adapter is None or
            adapter.__class__.__name__ != "ExplicitMotionPointTemporalAdapter"):
        return None
    return adapter


def _temporal_panel(spatial_map, image, title, lower, upper):
    """Render one temporal map with a clip-consistent color range."""
    height, width = image.shape[:2]
    spatial_map = cv2.resize(
        np.asarray(spatial_map, dtype=np.float32), (width, height),
        interpolation=cv2.INTER_LINEAR)
    lower, upper = float(lower), float(upper)
    normalized = np.clip(
        (spatial_map - lower) / max(upper - lower, 1e-8), 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.48, colored, 0.52, 0.0)
    panel = cv2.copyMakeBorder(
        overlay, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(
        panel, f"{title} scale=[{lower:.3g},{upper:.3g}]",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _save_point_temporal_debug(output_root, stem, model_input, core_model):
    """Export raw-frame and explicit motion-gate diagnostics for one clip."""
    adapter = _point_temporal_adapter(core_model)
    debug = getattr(adapter, "last_debug", None) if adapter is not None else None
    if not isinstance(debug, dict) or not debug:
        return 0

    seq_len = int(debug.get("seq_len", getattr(adapter, "seq_len", 1)))
    if seq_len <= 1 or model_input.shape[0] < seq_len:
        return 0
    clip = model_input[-seq_len:].detach().float().cpu().clamp(0.0, 1.0)
    clip_rgb = clip.repeat(1, 3, 1, 1) if clip.shape[1] == 1 else clip[:, :3]
    clip_images = (
        clip_rgb.permute(0, 2, 3, 1).numpy()[:, :, :, ::-1] * 255.0
    ).round().astype(np.uint8)
    preview_side = 256
    preview_images = [
        cv2.resize(image, (preview_side, preview_side),
                   interpolation=cv2.INTER_AREA)
        for image in clip_images]

    gray = clip.mean(dim=1).numpy()
    raw_abs_diff = np.zeros_like(gray, dtype=np.float32)
    raw_positive_diff = np.zeros_like(gray, dtype=np.float32)
    for time_index in range(1, seq_len):
        background = np.median(gray[:time_index], axis=0)
        residual = gray[time_index] - background
        raw_abs_diff[time_index] = np.abs(residual)
        raw_positive_diff[time_index] = np.maximum(residual, 0.0)

    arrays = {
        "raw_abs_diff": raw_abs_diff,
        "raw_positive_diff": raw_positive_diff,
    }
    debug_keys = (
        "aligned_abs_residual",
        "positive_residual",
        "snr",
        "instant_motion",
        "track_support",
        "learned_gate",
        "deterministic_gate",
        "final_gate",
    )
    for key in debug_keys:
        value = debug.get(key)
        if not torch.is_tensor(value) or value.shape[0] < seq_len:
            continue
        array = value[-seq_len:].detach().float().cpu().numpy()
        while array.ndim > 3:
            array = np.nanmean(array, axis=1)
        if array.ndim == 3:
            arrays[key] = np.nan_to_num(
                array, nan=0.0, posinf=0.0, neginf=0.0)

    titles = {
        "raw_abs_diff": "raw |current-history median|",
        "raw_positive_diff": "raw positive difference",
        "aligned_abs_residual": "aligned feature |residual|",
        "positive_residual": "positive feature residual",
        "snr": "feature residual SNR",
        "instant_motion": "instant deterministic motion",
        "track_support": "multi-frame trajectory support",
        "learned_gate": "learned gate",
        "deterministic_gate": "deterministic gate",
        "final_gate": "final learned/deterministic gate",
    }
    probability_keys = {
        "instant_motion", "track_support", "learned_gate",
        "deterministic_gate", "final_gate"}
    rows, stats_rows = [], []
    for key, array in arrays.items():
        finite = array[np.isfinite(array)]
        if key in probability_keys:
            lower, upper = 0.0, 1.0
        elif key == "snr":
            lower, upper = 0.0, 20.0
        else:
            lower = 0.0
            upper = max(float(np.percentile(finite, 99.5)), 1e-6)
        panels = []
        for time_index in range(seq_len):
            panels.append(_temporal_panel(
                array[time_index], preview_images[time_index],
                f"{titles.get(key, key)} t={time_index + 1}",
                lower, upper))
            values = np.asarray(array[time_index], dtype=np.float32)
            stats_rows.append({
                "map": key,
                "time": time_index + 1,
                "mean": float(values.mean()),
                "p95": float(np.percentile(values, 95.0)),
                "p99": float(np.percentile(values, 99.0)),
                "max": float(values.max()),
            })
        rows.append(np.hstack(panels))

    frame_dir = Path(output_root) / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        _save_jpg(
            frame_dir / "temporal_sequence.jpg", np.vstack(rows), quality=90)
    np.savez_compressed(frame_dir / "temporal_maps.npz", **arrays)
    with open(frame_dir / "temporal_stats.csv", "w", newline="",
              encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("map", "time", "mean", "p95", "p99", "max"))
        writer.writeheader()
        writer.writerows(stats_rows)
    return 3


def _directional_radius_panel(record, directions, level_name, index):
    """Draw base/support/final directional radii for one selected proposal."""
    canvas_size = 520
    margin = 58
    center = np.array([canvas_size // 2, canvas_size // 2 + 10],
                      dtype=np.float32)
    canvas = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)

    angles = np.asarray(directions, dtype=np.float32).reshape(-1)
    base = np.asarray(record["base_radius"], dtype=np.float32)
    support = np.asarray(record["support_radius"], dtype=np.float32)
    final = np.asarray(record["final_radius"], dtype=np.float32)
    if base.ndim > 1:
        base, support, final = base[index], support[index], final[index]
    maximum = max(float(np.nanmax(final)), float(np.nanmax(support)), 1e-6)
    radius_scale = (canvas_size * 0.5 - margin) / maximum
    direction_xy = np.stack((np.cos(angles), np.sin(angles)), axis=1)

    def polygon(radius):
        points = center[None, :] + direction_xy * radius[:, None] * radius_scale
        return np.round(points).astype(np.int32).reshape(-1, 1, 2)

    for fraction in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(
            canvas, tuple(center.astype(np.int32)),
            int((canvas_size * 0.5 - margin) * fraction),
            (215, 215, 215), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [polygon(support)], True, (0, 190, 255), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [polygon(base)], True, (255, 170, 0), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [polygon(final)], True, (0, 170, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, tuple(center.astype(np.int32)), 3, (0, 0, 0), -1)

    score = float(np.asarray(record["scores"])[index])
    blend = float(np.asarray(record["blend"])[index])
    proposal_center = np.asarray(record["centers"])[index]
    cv2.putText(
        canvas,
        f"{level_name} candidate {index}: score={score:.3f} blend={blend:.3f}",
        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20),
        1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"feature center=({proposal_center[0]:.2f}, {proposal_center[1]:.2f})",
        (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20),
        1, cv2.LINE_AA)
    if "predicted_centers" in record and "point_centers" in record:
        predicted_center = np.asarray(record["predicted_centers"])[index]
        point_center = np.asarray(record["point_centers"])[index]
        center_blend = float(np.asarray(record.get(
            "center_blend", np.ones(len(record["scores"]))))[index])
        cv2.putText(
            canvas,
            f"FT center=({predicted_center[0]:.2f},{predicted_center[1]:.2f}) "
            f"point=({point_center[0]:.2f},{point_center[1]:.2f}) "
            f"alpha={center_blend:.2f}",
            (12, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20),
            1, cv2.LINE_AA)
    if "assigned_peak_count" in record:
        peak_count = int(round(float(np.asarray(
            record["assigned_peak_count"])[index])))
        assignment_radius = float(np.asarray(
            record.get(
                "point_assignment_radius",
                np.zeros(len(record["scores"]), dtype=np.float32)))[index])
        discrete_shift = float(np.asarray(
            record.get(
                "discrete_center_shift",
                np.zeros(len(record["scores"]), dtype=np.float32)))[index])
        cv2.putText(
            canvas,
            f"assigned peaks={peak_count} ROI={assignment_radius:.2f} "
            f"center shift={discrete_shift:.2f}",
            (12, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20),
            1, cv2.LINE_AA)
    if "harmonic_gate" in record:
        harmonic_gate = np.asarray(
            record["harmonic_gate"], dtype=np.float32)
        if harmonic_gate.ndim > 1:
            harmonic_gate = harmonic_gate[index]
        gate_text = ",".join(
            f"{value:.2f}" for value in harmonic_gate[:8])
        cv2.putText(
            canvas, f"per-order gate=[{gate_text}]",
            (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
            (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, "blue: base Fourier", (12, canvas_size - 49),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, "yellow: point-supported reach", (12, canvas_size - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 150, 220), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, "green: final DFT envelope", (12, canvas_size - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 150, 0), 1, cv2.LINE_AA)
    return canvas


def _save_candidate_geometry_debug(output_root, stem, image, batch_index, device):
    """Export exact arrays and visual diagnostics for all Fourier levels."""
    store = GLOBAL_CANDIDATE_GEOMETRY_DICT.get(device)
    if not isinstance(store, dict) or not store:
        return 0

    frame_dir = Path(output_root) / stem
    frame_dir.mkdir(parents=True, exist_ok=True)


    max_preview_side = 512
    preview_scale = min(
        1.0, max_preview_side / max(float(max(image.shape[:2])), 1.0))
    if preview_scale < 1.0:
        preview_image = cv2.resize(
            image, None, fx=preview_scale, fy=preview_scale,
            interpolation=cv2.INTER_AREA)
    else:
        preview_image = image
    saved = 0
    ordered = sorted(store.items(), key=lambda item: item[0][0] * item[0][1],
                     reverse=True)
    map_keys = (
        ("support_prob", "point support probability"),
        ("filtered_support_prob", "robust filtered point support"),
        ("support_reliability", "local support reliability"),
        ("support_local_mass", "local support mass"),
        ("discrete_peak_map", "discrete predicted point peaks"),
        ("discrete_reference_peak_map", "native point-head reference peaks"),
        ("assigned_peak_map", "candidate-assigned point peaks"),
        ("assigned_peak_count", "assigned peak count per candidate"),
        ("discrete_center_shift", "discrete extent center correction"),
        ("candidate_prob", "Fourier proposal probability"),
        ("point_candidate_prob", "point-set candidate rescue"),
        ("joint_candidate_prob", "joint Fourier/point candidate"),
        ("expansion_mean", "directional expansion mean"),
        ("expansion_max", "directional expansion maximum"),
        ("spectral_active_point", "CPSI differentiable point evidence"),
        ("spectral_soft_assignment", "CPSI candidate soft assignment"),
        ("spectral_reliability", "CPSI candidate reliability"),
        ("spectral_harmonic_gate", "CPSI mean harmonic gate"),
        ("spectral_expansion", "CPSI directional expansion"),
    )
    bridge_keys = (
        ("fourier_candidate_prob", "native Fourier candidate probability"),
        ("confidence", "point confidence"),
        ("middle_ring", "middle-ring evidence"),
        ("outer_ring", "outer-ring evidence"),
        ("cluster_mass", "local point-set mass"),
        ("cluster_score", "point-set cluster score"),
        ("point_candidate_prob", "sparse point candidate"),
        ("joint_candidate_prob", "joint candidate probability"),
        ("hybrid_selected_scale", "selected Fourier pyramid scale"),
        ("accepted_centers", "accepted point centroids"),
        ("center_valid", "centroid assignment mask"),
        ("center_shift", "centroid correction magnitude"),
        ("center_x", "predicted centroid x"),
        ("center_y", "predicted centroid y"),
        ("gate", "learned bridge gate"),
        ("feature_in_energy", "Fourier feature energy before bridge"),
        ("delta_energy", "bridge residual energy"),
        ("feature_out_energy", "Fourier feature energy after bridge"),
    )

    for level_index, (resolution, entry) in enumerate(ordered):
        level_name = f"P{level_index + 3}_{resolution[0]}x{resolution[1]}"
        panels, arrays = [], {
            "oracle_mode": np.asarray(
                bool(entry.get("oracle_mode", False)), dtype=np.uint8),
            "point_mode": np.asarray(
                str(entry.get("point_mode", "soft"))),
        }
        for key, title in map_keys:
            spatial = _debug_spatial_map(entry.get(key), batch_index)
            if spatial is not None:
                arrays[key] = spatial
            panels.append(_heatmap_panel(
                spatial, preview_image, f"{level_name} {title}"))

        bridge_debug = entry.get("bridge_debug", {})
        for key, title in bridge_keys:
            spatial = _debug_spatial_map(
                bridge_debug.get(key) if isinstance(bridge_debug, dict) else None,
                batch_index)
            if spatial is not None:
                arrays[f"bridge_{key}"] = spatial
            panels.append(_heatmap_panel(
                spatial, preview_image, f"{level_name} {title}"))

        panel_height, panel_width = panels[0].shape[:2]
        columns = 3
        rows = (len(panels) + columns - 1) // columns
        blank = np.zeros_like(panels[0])
        panels.extend([blank] * (rows * columns - len(panels)))
        mosaic = np.vstack([
            np.hstack(panels[row * columns:(row + 1) * columns])
            for row in range(rows)
        ])
        _save_jpg(frame_dir / f"{level_name}_maps.jpg", mosaic, quality=88)
        np.savez_compressed(frame_dir / f"{level_name}_maps.npz", **arrays)
        saved += 2

        directions = _debug_tensor_for_frame(entry.get("directions"), 0)
        records = entry.get("topk_records", [])
        if directions is None or not isinstance(records, list):
            continue
        available_batches = sorted({
            int(record.get("batch", 0)) for record in records
            if isinstance(record, dict)
        })
        if not available_batches:
            continue
        record_batch = min(
            available_batches, key=lambda value: abs(value - int(batch_index)))
        candidate_index = 0
        for record in records:
            if (not isinstance(record, dict) or
                    int(record.get("batch", 0)) != record_batch):
                continue
            scores_value = record["scores"]
            count = (
                int(scores_value.numel())
                if torch.is_tensor(scores_value)
                else int(np.asarray(scores_value).size)
            )
            converted = {
                key: (
                    value.detach().float().cpu().numpy()
                    if torch.is_tensor(value) else value)
                for key, value in record.items()
                if key != "batch"
            }
            for local_index in range(count):
                panel = _directional_radius_panel(
                    converted, directions, level_name, local_index)
                _save_png(
                    frame_dir /
                    f"{level_name}_candidate_{candidate_index:02d}_direction.png",
                    panel)
                np.savez_compressed(
                    frame_dir /
                    f"{level_name}_candidate_{candidate_index:02d}_direction.npz",
                    flat_id=np.asarray(
                        converted["flat_ids"][local_index], dtype=np.int64),
                    directions=np.asarray(directions, dtype=np.float32),
                    center=np.asarray(
                        converted["centers"][local_index], dtype=np.float32),
                    predicted_center=np.asarray(
                        converted.get(
                            "predicted_centers",
                            converted["centers"])[local_index],
                        dtype=np.float32),
                    point_center=np.asarray(
                        converted.get(
                            "point_centers",
                            converted["centers"])[local_index],
                        dtype=np.float32),
                    point_extent_center=np.asarray(
                        converted.get(
                            "point_extent_centers",
                            converted.get(
                                "point_centers",
                                converted["centers"]))[local_index],
                        dtype=np.float32),
                    center_blend=np.asarray(
                        converted.get(
                            "center_blend",
                            np.ones(count, dtype=np.float32))[local_index],
                        dtype=np.float32),
                    discrete_center_blend=np.asarray(
                        converted.get(
                            "discrete_center_blend",
                            np.zeros(count, dtype=np.float32))[local_index],
                        dtype=np.float32),
                    discrete_center_shift=np.asarray(
                        converted.get(
                            "discrete_center_shift",
                            np.zeros(count, dtype=np.float32))[local_index],
                        dtype=np.float32),
                    point_assignment_radius=np.asarray(
                        converted.get(
                            "point_assignment_radius",
                            np.zeros(count, dtype=np.float32))[local_index],
                        dtype=np.float32),
                    assigned_peak_count=np.asarray(
                        converted.get(
                            "assigned_peak_count",
                            np.zeros(count, dtype=np.float32))[local_index],
                        dtype=np.float32),
                    assigned_peak_xy=np.asarray(
                        converted.get(
                            "assigned_peak_xy",
                            np.empty((count, 0, 2),
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                    assigned_peak_scores=np.asarray(
                        converted.get(
                            "assigned_peak_scores",
                            np.empty((count, 0),
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                    score=np.asarray(
                        converted["scores"][local_index], dtype=np.float32),
                    blend=np.asarray(
                        converted["blend"][local_index], dtype=np.float32),
                    base_radius=np.asarray(
                        converted["base_radius"][local_index],
                        dtype=np.float32),
                    support_radius=np.asarray(
                        converted["support_radius"][local_index],
                        dtype=np.float32),
                    point_envelope=np.asarray(
                        converted.get(
                            "point_envelope",
                            converted["support_radius"])[local_index],
                        dtype=np.float32),
                    raw_expansion=np.asarray(
                        converted.get(
                            "raw_expansion",
                            converted["expansion"])[local_index],
                        dtype=np.float32),
                    expansion=np.asarray(
                        converted["expansion"][local_index],
                        dtype=np.float32),
                    final_radius=np.asarray(
                        converted["final_radius"][local_index],
                        dtype=np.float32),
                    base_coefficients=np.asarray(
                        converted["base_coefficients"][local_index],
                        dtype=np.float32),
                    refined_coefficients=np.asarray(
                        converted["refined_coefficients"][local_index],
                        dtype=np.float32),
                    harmonic_gate=np.asarray(
                        converted.get(
                            "harmonic_gate",
                            np.zeros((count, 0),
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                    point_mass=np.asarray(
                        converted.get(
                            "point_mass",
                            np.zeros(count,
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                    reliability=np.asarray(
                        converted.get(
                            "reliability",
                            np.zeros(count,
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                    roi_radius=np.asarray(
                        converted.get(
                            "roi_radius",
                            np.zeros(count,
                                     dtype=np.float32))[local_index],
                        dtype=np.float32),
                )
                candidate_index += 1
                saved += 2
    return saved


def _decode_raw_point_head(point_preds, batch_index, input_shape, output_shape,
                           conf_threshold=0.05, max_points=1000):
    """Decode BioCountInfer peaks without macro-contour filtering or clustering."""
    if not isinstance(point_preds, dict) or "pred_logits" not in point_preds:
        return [], None

    logits_all = point_preds["pred_logits"]
    if logits_all.ndim != 4 or logits_all.shape[0] == 0:
        return [], None
    index = batch_index if batch_index < logits_all.shape[0] else logits_all.shape[0] - 1
    logits = logits_all[index, 0].float()
    probability = logits.sigmoid()


    local_max = F.max_pool2d(probability[None, None], 3, stride=1, padding=1)[0, 0]
    peak_mask = (probability >= float(conf_threshold)) & (probability == local_max)
    peak_y, peak_x = torch.where(peak_mask)
    if peak_x.numel() == 0:
        return [], probability.detach().cpu().numpy()

    scores = probability[peak_y, peak_x]
    if max_points > 0 and scores.numel() > max_points:
        scores, keep = scores.topk(max_points)
        peak_y, peak_x = peak_y[keep], peak_x[keep]
    else:
        order = scores.argsort(descending=True)
        scores, peak_y, peak_x = scores[order], peak_y[order], peak_x[order]

    map_h, map_w = probability.shape
    input_h, input_w = int(input_shape[0]), int(input_shape[1])
    offsets_all = point_preds.get("pred_offsets")
    wh_all = point_preds.get("pred_wh")
    if offsets_all is not None:
        offsets = offsets_all[index, :, peak_y, peak_x].float().transpose(0, 1)
    else:
        offsets = torch.full((peak_x.numel(), 2), 0.5, device=peak_x.device)
    if wh_all is not None:
        wh_cells = wh_all[index, :, peak_y, peak_x].float().transpose(0, 1)
    else:
        wh_cells = torch.zeros((peak_x.numel(), 2), device=peak_x.device)

    centers_map = torch.stack((peak_x.float(), peak_y.float()), dim=1) + offsets
    centers_input = centers_map * centers_map.new_tensor([input_w / map_w, input_h / map_h])
    wh_input = wh_cells * wh_cells.new_tensor([input_w / map_w, input_h / map_h])

    wh_input = wh_input.clamp_min(1.0)
    boxes = torch.cat((centers_input - wh_input / 2, centers_input + wh_input / 2), dim=1)
    boxes = scale_coords((input_h, input_w), boxes, output_shape).round()

    height, width = output_shape[:2]
    records = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        x1, x2 = sorted((max(0.0, min(x1, width - 1.0)), max(0.0, min(x2, width - 1.0))))
        y1, y2 = sorted((max(0.0, min(y1, height - 1.0)), max(0.0, min(y2, height - 1.0))))
        records.append((x1, y1, x2, y2, float(score.item())))
    return records, probability.detach().cpu().numpy()


def _filter_points_near_fourier_centers(records, polygons, radius_scale=1.75,
                                        min_radius=20.0):
    """Keep final point outputs close to at least one retained Fourier candidate.

    This is deliberately an inference-only output filter. It is applied after
    the point head has been decoded and after Fourier polygon NMS, so it cannot
    change the point heatmap, Fourier coefficients, model features, or weights.
    The acceptance radius adapts to each predicted contour's radial extent.
    """
    if not records or not polygons:
        return []

    radius_scale = float(radius_scale)
    min_radius = float(min_radius)
    supports = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] == 0 or not np.isfinite(points).all():
            continue
        center = points.mean(axis=0)
        contour_radius = float(np.linalg.norm(points - center[None], axis=1).max())
        radius = max(min_radius, radius_scale * contour_radius)
        supports.append((center, radius * radius))

    if not supports:
        return []

    kept = []
    for record in records:
        x1, y1, x2, y2, _ = record
        point = np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)
        if any(float(np.sum((point - center) ** 2)) <= radius_sq
               for center, radius_sq in supports):
            kept.append(record)
    return kept


def _draw_raw_points(image, records, conf_threshold, draw_scores=False,
                     raw_count=None):
    for x1, y1, x2, y2, score in records:
        center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(image, center, (255, 0, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        if draw_scores:
            _draw_text(image, f"{score:.2f}", (center[0] + 4, center[1] - 4),
                       (255, 0, 255), 0.35)
    if raw_count is None:
        hud = f"Raw BioCountInfer peaks={len(records)} threshold={conf_threshold:.3f}"
    else:
        hud = (f"Center-filtered BioCountInfer peaks={len(records)}/{int(raw_count)} "
               f"threshold={conf_threshold:.3f}")
    cv2.rectangle(image, (5, 5), (min(image.shape[1] - 1, 500), 34), (0, 0, 0), -1)
    cv2.putText(image, hud, (11, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_raw_yolo(image, detections, names, line_thickness=2):
    for row in detections:
        x1, y1, x2, y2, confidence, class_id = row[:6]
        cls = int(class_id.item())
        class_name = names[cls] if not isinstance(names, dict) else names.get(cls, str(cls))
        color = (0, 165, 255)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, line_thickness, cv2.LINE_AA)
        _draw_text(image, f"raw YOLO {class_name} {float(confidence.item()):.2f}",
                   (int(x1), int(y1) - 5), color, 0.42)
    hud = f"Raw Detect boxes after standard NMS={len(detections)}"
    cv2.rectangle(image, (5, 5), (min(image.shape[1] - 1, 470), 34), (0, 0, 0), -1)
    cv2.putText(image, hud, (11, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_fourier_proposals(image, polygons, confidences, line_thickness=2):
    """Visualize the no-Detect candidate path without inventing YOLO boxes."""
    for polygon, confidence in zip(polygons, confidences):
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True,
                      (255, 255, 0), line_thickness, cv2.LINE_AA)
        center = np.round(polygon.mean(axis=0)).astype(np.int32)
        cv2.circle(image, tuple(center), 3, (0, 165, 255), -1, cv2.LINE_AA)
        _draw_text(image, f"Fourier proposal {confidence:.2f}",
                   (int(center[0]), int(center[1]) - 5),
                   (0, 165, 255), 0.42)
    hud = f"Fourier proposals after polygon NMS={len(polygons)} (no Detect head)"
    cv2.rectangle(image, (5, 5), (min(image.shape[1] - 1, 610), 34),
                  (0, 0, 0), -1)
    cv2.putText(image, hud, (11, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _point_heatmap_overlay(image, probability):
    if probability is None:
        return image
    heatmap = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap(np.round(heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(image, 0.55, colored, 0.45, 0.0)


def _save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode result: {path}")
    encoded.tofile(str(path))


def _save_jpg(path, image, quality=90):
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode result: {path}")
    encoded.tofile(str(path))


_PAPER_VIS_DEFAULTS = {
    "feature_layers": [4, 17, 20, 23, 27, 30, 33, 36, 37, 38],
    "layer_labels": {
        "4": "Backbone shallow P3 (layer 4)",
        "17": "Shared neck P3 (layer 17)",
        "20": "Shared neck P4 (layer 20)",
        "23": "Shared neck P5 (layer 23)",
        "27": "Fourier P3 before bridge (layer 27)",
        "30": "Fourier P4 before bridge (layer 30)",
        "33": "Fourier P5 before bridge (layer 33)",
        "36": "Fourier P3 after bridge (layer 36)",
        "37": "Fourier P4 after bridge (layer 37)",
        "38": "Fourier P5 after bridge (layer 38)",
    },
    "feature_scales": {},
    "overview_columns": 5,
    "save_individual_panels": True,
    "jpeg_quality": 95,
}


def _load_paper_vis_config(path):
    """Load visualization-only settings without mutating the model YAML."""
    config = {
        **_PAPER_VIS_DEFAULTS,
        "layer_labels": dict(_PAPER_VIS_DEFAULTS["layer_labels"]),
        "feature_scales": {},
    }
    if not path:
        return config
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Paper visualization config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as stream:
        user_config = json.load(stream)
    if not isinstance(user_config, dict):
        raise ValueError("Paper visualization config must contain a JSON object")
    config.update(user_config)
    config["layer_labels"] = {
        **_PAPER_VIS_DEFAULTS["layer_labels"],
        **{str(key): str(value) for key, value in
           dict(user_config.get("layer_labels", {})).items()},
    }
    config["feature_layers"] = [
        int(value) for value in config.get("feature_layers", [])]
    config["overview_columns"] = max(
        int(config.get("overview_columns", 5)), 1)
    config["jpeg_quality"] = min(max(
        int(config.get("jpeg_quality", 95)), 1), 100)
    return config


def _feature_energy(tensor):
    """Compress BCHW features to batch-preserving RMS activation maps."""
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        return None
    with torch.no_grad():
        return tensor.detach().float().square().mean(
            dim=1).sqrt().cpu().numpy().astype(np.float32)


class _PaperFeatureHooks:
    """Inference-only hooks for publication feature maps.

    Only two-dimensional channel-reduced maps are retained. Hooks neither
    replace module inputs/outputs nor add trainable state to the checkpoint.
    """

    def __init__(self, core_model, layer_indices):
        self.maps = {}
        self._handles = []
        self._ft_inputs = {}
        layers = getattr(core_model, "model", None)
        if layers is None:
            raise RuntimeError("Loaded model does not expose its sequential layers")
        for layer_index in layer_indices:
            if layer_index < 0 or layer_index >= len(layers):
                print(f"WARNING: paper feature layer {layer_index} is out of range")
                continue
            handle = layers[layer_index].register_forward_hook(
                self._layer_hook(layer_index))
            self._handles.append(handle)

        ft_head = (
            core_model.get_module_byname("FTInfer")
            if hasattr(core_model, "get_module_byname") else None)
        adapters = getattr(ft_head, "temporal_adapters", ())
        for level_index, adapter in enumerate(adapters):
            self._handles.append(adapter.register_forward_pre_hook(
                self._ft_pre_hook(level_index)))
            self._handles.append(adapter.register_forward_hook(
                self._ft_post_hook(level_index)))

    @staticmethod
    def _first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if torch.is_tensor(item):
                    return item
        return None

    def _layer_hook(self, layer_index):
        def hook(_module, _inputs, output):
            tensor = self._first_tensor(output)
            energy = _feature_energy(tensor)
            if energy is not None:
                self.maps[f"layer_{layer_index}_energy"] = energy
        return hook

    def _ft_pre_hook(self, level_index):
        def hook(_module, inputs):
            tensor = self._first_tensor(inputs)
            if tensor is not None:
                self._ft_inputs[level_index] = tensor.detach()
                energy = _feature_energy(tensor)
                if energy is not None:
                    self.maps[f"ft_temporal_P{level_index + 3}_input_energy"] = energy
        return hook

    def _ft_post_hook(self, level_index):
        def hook(_module, _inputs, output):
            tensor = self._first_tensor(output)
            before = self._ft_inputs.pop(level_index, None)
            energy = _feature_energy(tensor)
            if energy is not None:
                self.maps[f"ft_temporal_P{level_index + 3}_output_energy"] = energy
            if (tensor is not None and before is not None and
                    tensor.shape == before.shape):
                delta = _feature_energy(tensor.detach() - before)
                if delta is not None:
                    self.maps[f"ft_temporal_P{level_index + 3}_delta_energy"] = delta
        return hook

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._ft_inputs.clear()


def _array_for_batch(value, batch_index):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim >= 3:
        index = min(max(int(batch_index), 0), array.shape[0] - 1)
        array = array[index]
    while array.ndim > 2:
        array = np.nanmax(array, axis=0)
    if array.ndim != 2:
        return None
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _paper_scale(name, spatial_map, config, probability=False, signed=False):
    configured = dict(config.get("feature_scales", {})).get(name)
    if isinstance(configured, (tuple, list)) and len(configured) == 2:
        return float(configured[0]), float(configured[1])
    if probability:
        return 0.0, 1.0
    finite = spatial_map[np.isfinite(spatial_map)]
    if not finite.size:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        magnitude = max(float(np.percentile(np.abs(finite), 99.5)), 1e-6)
        return -magnitude, magnitude
    return 0.0, max(float(np.percentile(finite, 99.5)), 1e-6)


def _paper_heatmap_panel(spatial_map, image, title, name, config,
                         probability=False, signed=False):
    height, width = image.shape[:2]
    if spatial_map is None:
        spatial_map = np.zeros((height, width), dtype=np.float32)
    spatial_map = np.asarray(spatial_map, dtype=np.float32)
    lower, upper = _paper_scale(
        name, spatial_map, config, probability=probability, signed=signed)
    resized = cv2.resize(
        spatial_map, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(
        (resized - lower) / max(upper - lower, 1e-8), 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.48, colored, 0.52, 0.0)
    return _paper_image_panel(
        overlay, f"{title}  scale=[{lower:.3g},{upper:.3g}]")


def _paper_image_panel(image, title):
    panel = cv2.copyMakeBorder(
        image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(panel, title, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _paper_qualitative_panel(image, title="Final Fourier contour"):
    """Create the final qualitative panel with a 22 pt paper title."""
    foreground_thickness = max(1, int(round(image.shape[1] / 700.0)))
    scale = _paper_font_scale(image.shape[1], 22, 5.55,
                              foreground_thickness)
    _, text_height = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, scale,
        foreground_thickness)[0]
    padding = max(10, int(round(image.shape[1] / 80.0)))
    bar_height = text_height + 2 * padding
    panel = cv2.copyMakeBorder(
        image, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(panel, title, (padding, padding + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                foreground_thickness, cv2.LINE_AA)
    return panel


def _sequence_panel(model_input, image, seq_len):
    """Place the causal input clip in one image-sized overview panel."""
    clip = model_input[-seq_len:].detach().float().cpu().clamp(0.0, 1.0)
    if clip.shape[1] == 1:
        clip = clip.repeat(1, 3, 1, 1)
    else:
        clip = clip[:, :3]
    frames = (clip.permute(0, 2, 3, 1).numpy()[:, :, :, ::-1] * 255.0)
    frames = np.round(frames).astype(np.uint8)
    height, width = image.shape[:2]
    columns = 3
    rows = 2
    cell_w, cell_h = max(width // columns, 1), max(height // rows, 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames[-columns * rows:]):
        row, column = divmod(index, columns)
        thumb = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y0, x0 = row * cell_h, column * cell_w
        canvas[y0:min(y0 + cell_h, height), x0:min(x0 + cell_w, width)] =\
            thumb[:min(cell_h, height - y0), :min(cell_w, width - x0)]
        cv2.putText(canvas, f"t-{len(frames) - index - 1}",
                    (x0 + 6, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return _paper_image_panel(canvas, "Causal input sequence")


def _paper_geometry_maps(device, batch_index):
    """Select the high-resolution P3 diagnostics used in the main flow."""
    store = GLOBAL_CANDIDATE_GEOMETRY_DICT.get(device)
    if not isinstance(store, dict) or not store:
        return {}
    resolution, entry = max(
        store.items(), key=lambda item: int(item[0][0]) * int(item[0][1]))
    if not isinstance(entry, dict):
        return {}
    bridge = entry.get("bridge_debug", {})
    bridge = bridge if isinstance(bridge, dict) else {}

    def pick(*candidates):
        for source, key in candidates:
            value = source.get(key)
            spatial = _debug_spatial_map(value, batch_index)
            if spatial is not None:
                return spatial
        return None

    return {
        "native_candidate": pick(
            (bridge, "fourier_candidate_prob"), (entry, "candidate_prob")),
        "point_candidate": pick(
            (bridge, "point_candidate_prob"), (entry, "point_candidate_prob")),
        "joint_candidate": pick(
            (bridge, "joint_candidate_prob"), (entry, "joint_candidate_prob")),
        "selected_scale": pick((bridge, "hybrid_selected_scale")),
        "bridge_gate": pick((bridge, "gate")),
        "bridge_delta": pick((bridge, "delta_energy")),
        "assigned_peaks": pick((entry, "assigned_peak_map")),
        "expansion": pick((entry, "expansion_max"), (entry, "expansion_mean")),
        "resolution": np.asarray(resolution, dtype=np.int32),
    }


def _save_paper_flow(output_root, stem, image, final_image, point_image,
                     point_probability, model_input, core_model, hooks,
                     batch_index, seq_len, config, plain_final=False):
    """Save a curated input-to-contour figure and its exact 2-D arrays."""
    frame_dir = Path(output_root) / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def hook_map(name):
        value = hooks.maps.get(name) if hooks is not None else None
        spatial = _array_for_batch(value, batch_index)
        if spatial is not None:
            arrays[name] = spatial
        return spatial

    point_feature = GLOBAL_POINT_FEATURE_DICT.get(model_input.device)
    point_feature = (
        point_feature.get("feature") if isinstance(point_feature, dict) else None)
    point_energy_all = _feature_energy(point_feature)
    point_energy = _array_for_batch(point_energy_all, batch_index)
    if point_energy is not None:
        arrays["point_decoder_energy"] = point_energy

    point_head = (
        core_model.get_module_byname("BioCountInfer")
        if hasattr(core_model, "get_module_byname") else None)
    adapter = getattr(point_head, "temporal_adapter", None)
    temporal_debug = getattr(adapter, "last_debug", {})
    final_gate = _debug_spatial_map(
        temporal_debug.get("final_gate")
        if isinstance(temporal_debug, dict) else None, batch_index)
    if final_gate is not None:
        arrays["point_final_gate"] = final_gate
    if point_probability is not None:
        arrays["point_probability"] = np.asarray(
            point_probability, dtype=np.float32)

    geometry = _paper_geometry_maps(model_input.device, batch_index)
    for key, value in geometry.items():
        if key != "resolution" and value is not None:
            arrays[f"geometry_{key}"] = np.asarray(value, dtype=np.float32)


    captured_maps = {}
    if hooks is not None:
        for name, value in hooks.maps.items():
            spatial = _array_for_batch(value, batch_index)
            if spatial is not None:
                captured_maps[name] = spatial
                arrays[name] = spatial

    backbone_layer = int(config.get("backbone_layer", 4))
    backbone_name = f"layer_{backbone_layer}_energy"
    panels = [
        ("01_input_sequence", _sequence_panel(model_input, image, seq_len)),
        (f"02_backbone_l{backbone_layer}", _paper_heatmap_panel(
            hook_map(backbone_name), image, "Backbone shallow feature energy",
            backbone_name, config)),
        ("03_shared_neck_p3", _paper_heatmap_panel(
            hook_map("layer_17_energy"), image, "Shared neck P3 energy",
            "layer_17_energy", config)),
        ("04_point_temporal_gate", _paper_heatmap_panel(
            final_gate, image, "Point temporal final gate",
            "point_final_gate", config, probability=True)),
        ("05_point_decoder", _paper_heatmap_panel(
            point_energy, image, "Point decoder feature energy",
            "point_decoder_energy", config)),
        ("06_point_probability", _paper_heatmap_panel(
            point_probability, image, "BioCountInfer probability",
            "point_probability", config, probability=True)),
        ("07_point_peaks", _paper_image_panel(
            point_image, "Decoded point peaks")),
        ("08_fourier_p3_before_bridge", _paper_heatmap_panel(
            hook_map("layer_27_energy"), image, "Fourier P3 before bridge",
            "layer_27_energy", config)),
        ("09_native_candidate", _paper_heatmap_panel(
            geometry.get("native_candidate"), image,
            "Native Fourier candidate probability", "native_candidate",
            config, probability=True)),
        ("10_point_candidate", _paper_heatmap_panel(
            geometry.get("point_candidate"), image,
            "Point-set candidate rescue", "point_candidate",
            config, probability=True)),
        ("11_joint_candidate", _paper_heatmap_panel(
            geometry.get("joint_candidate"), image,
            "Hybrid joint candidate", "joint_candidate",
            config, probability=True)),
        ("12_bridge_delta", _paper_heatmap_panel(
            geometry.get("bridge_delta"), image,
            "Directional bridge residual energy", "bridge_delta", config)),
        ("13_ft_temporal_delta", _paper_heatmap_panel(
            hook_map("ft_temporal_P3_delta_energy"), image,
            "Fourier P3 temporal delta", "ft_temporal_P3_delta_energy",
            config)),
        ("14_directional_expansion", _paper_heatmap_panel(
            geometry.get("expansion"), image,
            "Point-supported directional expansion", "expansion", config)),
        ("15_final_contour", final_image.copy() if plain_final
         else _paper_qualitative_panel(final_image)),
    ]

    if bool(config.get("save_individual_panels", True)):
        for name, panel in panels:
            _save_png(frame_dir / f"{name}.png", panel)

    feature_panels = []
    layer_labels = dict(config.get("layer_labels", {}))
    ordered_feature_names = [
        f"layer_{index}_energy"
        for index in config.get("feature_layers", [])]
    ordered_feature_names.extend([
        f"ft_temporal_P{level}_{suffix}_energy"
        for level in (3, 4, 5)
        for suffix in ("input", "output", "delta")
    ])
    for name in ordered_feature_names:
        spatial = captured_maps.get(name)
        if spatial is None:
            continue
        if name.startswith("layer_"):
            layer_index = name.split("_")[1]
            title = layer_labels.get(layer_index, name)
        else:
            title = name.replace("_", " ")
        panel = _paper_heatmap_panel(
            spatial, image, title, name, config)
        feature_panels.append((name, panel))
    if feature_panels:
        feature_dir = frame_dir / "feature_maps"
        feature_dir.mkdir(parents=True, exist_ok=True)
        if bool(config.get("save_individual_panels", True)):
            for name, panel in feature_panels:
                _save_png(feature_dir / f"{name}.png", panel)
        feature_columns = min(5, max(len(feature_panels), 1))
        feature_rows = int(np.ceil(
            len(feature_panels) / float(feature_columns)))
        feature_blank = np.zeros_like(feature_panels[0][1])
        feature_images = [panel for _, panel in feature_panels]
        feature_images.extend(
            [feature_blank] *
            (feature_rows * feature_columns - len(feature_images)))
        feature_overview = np.vstack([
            np.hstack(feature_images[
                row * feature_columns:(row + 1) * feature_columns])
            for row in range(feature_rows)
        ])
        _save_jpg(
            frame_dir / "feature_pyramid_overview.jpg", feature_overview,
            quality=int(config.get("jpeg_quality", 95)))

    columns = int(config.get("overview_columns", 5))
    rows = int(np.ceil(len(panels) / float(columns)))
    blank = np.zeros_like(panels[0][1])
    overview_height, overview_width = blank.shape[:2]
    images = [
        panel if panel.shape[:2] == (overview_height, overview_width)
        else cv2.resize(panel, (overview_width, overview_height),
                        interpolation=cv2.INTER_AREA)
        for _, panel in panels
    ]
    images.extend([blank] * (rows * columns - len(images)))
    overview = np.vstack([
        np.hstack(images[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])
    _save_jpg(
        frame_dir / "paper_flow_overview.jpg", overview,
        quality=int(config.get("jpeg_quality", 95)))
    np.savez_compressed(frame_dir / "paper_flow_maps.npz", **arrays)
    with open(frame_dir / "paper_flow_stats.csv", "w", newline="",
              encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("map", "shape", "min", "mean", "p95", "p99", "max"))
        writer.writeheader()
        for name, value in arrays.items():
            values = np.asarray(value, dtype=np.float32)
            writer.writerow({
                "map": name,
                "shape": "x".join(str(size) for size in values.shape),
                "min": float(values.min()),
                "mean": float(values.mean()),
                "p95": float(np.percentile(values, 95.0)),
                "p99": float(np.percentile(values, 99.0)),
                "max": float(values.max()),
            })
    return frame_dir / "paper_flow_overview.jpg"


def _resolve_label_files(label_root, source_path, source_root):
    source_path = Path(source_path)
    relative_parent = Path()
    if source_root is not None:
        try:
            relative_parent = source_path.resolve().parent.relative_to(source_root.resolve())
        except ValueError:
            pass
    label_dir = Path(label_root) / relative_parent
    stems = [source_path.stem]
    if source_path.stem.lower().endswith("ir"):
        stems.append(source_path.stem[:-2])
    for stem in stems:
        if (label_dir / f"{stem}.ft").is_file():
            return label_dir / f"{stem}.ft", label_dir / f"single_{stem}.txt"
    stem = stems[0]
    return label_dir / f"{stem}.ft", label_dir / f"single_{stem}.txt"


def _oracle_targets_for_paths(frame_paths, label_root, source_root, device):
    """Build flattened [time, class(0=micro,1=macro), xywh] oracle targets."""
    rows = []
    for time_index, frame_path in enumerate(frame_paths):
        ft_path, single_path = _resolve_label_files(
            label_root, frame_path, source_root)
        macro_path = ft_path.with_suffix('.txt')
        for row in _read_rows(macro_path):
            if len(row) >= 5:
                rows.append([time_index, 1.0, *row[1:5]])
        for row in _read_rows(single_path):
            if len(row) >= 5:
                rows.append([time_index, 0.0, *row[1:5]])
    if not rows:
        return torch.zeros((0, 6), device=device, dtype=torch.float32)
    return torch.tensor(rows, device=device, dtype=torch.float32)


def _diagnostic_overlay(image, stem, pred_polygons, confidences, ft_path, single_path,
                        line_thickness=2, point_radius=3, paper_style=False):
    height, width = image.shape[:2]
    gt_polygons = [
        polygon for polygon in
        (_fourier_gt_polygon(row, width, height) for row in _read_rows(ft_path))
        if polygon is not None
    ]
    single_items = [
        item for item in
        (_yolo_single(row, width, height) for row in _read_rows(single_path))
        if item is not None
    ]
    matches = _greedy_polygon_matches(pred_polygons, gt_polygons, image.shape)


    gt_points = [[] for _ in gt_polygons]
    unassigned = []
    for center, box in single_items:
        owners = [index for index, polygon in enumerate(gt_polygons)
                  if _point_inside(polygon, center)]
        if owners:
            gt_points[owners[0]].append((center, box))
        else:
            unassigned.append((center, box))

    for polygon in gt_polygons:
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True,
                      (0, 0, 255), line_thickness, cv2.LINE_AA)

    rows = []
    covered_total = 0
    assigned_total = sum(len(points) for points in gt_points)
    matched_iou_sum = 0.0
    for pred_index, polygon in enumerate(pred_polygons):
        cv2.polylines(image, [polygon.reshape(-1, 1, 2)], True,
                      (255, 255, 0), line_thickness, cv2.LINE_AA)
        gt_index, iou = matches.get(pred_index, (-1, 0.0))
        points = gt_points[gt_index] if gt_index >= 0 else []
        covered = sum(_point_inside(polygon, center) for center, _ in points)
        coverage = covered / len(points) if points else 0.0
        covered_total += covered
        if gt_index >= 0:
            matched_iou_sum += iou
        anchor = polygon[polygon[:, 1].argmin()]
        confidence = confidences[pred_index]
        if paper_style:
            _draw_paper_text(
                image, f"P{pred_index}", (anchor[0], anchor[1] - 8),
                (255, 255, 0), point_size=16)
        else:
            _draw_text(image,
                       f"P{pred_index}->G{gt_index} conf={confidence:.2f} "
                       f"IoU={iou:.3f} cover={covered}/{len(points)} ({coverage:.1%})",
                       (anchor[0], anchor[1] - 5), (255, 255, 0))
        rows.append({
            "frame": stem,
            "prediction": pred_index,
            "matched_gt": gt_index,
            "confidence": confidence,
            "iou": iou,
            "gt_points": len(points),
            "covered_points": covered,
            "point_coverage": coverage,
        })

    matched_targets = {gt_index for gt_index, _ in matches.values()}
    for gt_index, points in enumerate(gt_points):
        if gt_index in matched_targets:
            continue
        rows.append({
            "frame": stem,
            "prediction": -1,
            "matched_gt": gt_index,
            "confidence": 0.0,
            "iou": 0.0,
            "gt_points": len(points),
            "covered_points": 0,
            "point_coverage": 0.0,
        })


    gt_to_pred = {gt_index: pred_index for pred_index, (gt_index, _) in matches.items()}
    for gt_index, points in enumerate(gt_points):
        pred_index = gt_to_pred.get(gt_index)
        pred_polygon = pred_polygons[pred_index] if pred_index is not None else None
        for center, box in points:
            inside = _point_inside(pred_polygon, center)
            color = (0, 220, 0) if inside else (0, 255, 255)
            box_thickness = max(1, line_thickness // 2)
            cv2.rectangle(image, box[:2], box[2:], color,
                          box_thickness, cv2.LINE_AA)
            cv2.circle(image, center, point_radius, color, -1, cv2.LINE_AA)
    for center, box in unassigned:
        box_thickness = max(1, line_thickness // 2)
        cv2.rectangle(image, box[:2], box[2:], (0, 128, 255),
                      box_thickness, cv2.LINE_AA)
        cv2.circle(image, center, point_radius, (0, 128, 255), -1, cv2.LINE_AA)

    mean_iou = matched_iou_sum / len(gt_polygons) if gt_polygons else 0.0
    total_coverage = covered_total / assigned_total if assigned_total else 0.0
    hud = (f"{stem} | GT:red Pred:cyan covered:green missed:yellow | "
           f"GT={len(gt_polygons)} Pred={len(pred_polygons)} "
           f"mIoU={mean_iou:.3f} point-cover={covered_total}/{assigned_total} ({total_coverage:.1%})")
    if not paper_style:
        cv2.rectangle(image, (5, 5), (min(width - 1, 930), 36),
                      (0, 0, 0), -1)
        cv2.putText(image, hud, (11, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return rows


@torch.no_grad()
def run(weights=ROOT / "weights/best.pt",
        source=ROOT / "data/images",
        imgsz=(640, 640),
        conf_thres=0.15,
        iou_thres=0.45,
        device="",
        project=ROOT / "runs/detect_ft_raw",
        name="exp",
        exist_ok=False,
        half=True,
        augment=False,
        agnostic_nms=False,
        nms_polygon=1,
        line_thickness=2,
        draw_box=False,
        save_txt=True,
        nosave=False,
        diagnostic=True,
        gt_label_dir="",
        point_radius=3,
        point_conf_thres=0.60,
        point_max_detections=100,
        draw_point_scores=False,
        point_center_filter=True,
        point_center_radius_scale=1.75,
        point_center_min_radius=20.0,
        oracle_gt=False,
        geometry_debug=True,
        temporal_debug=False,
        temporal_debug_limit=0,
        paper_flow=False,
         paper_flow_limit=0,
         paper_flow_frame_stems="",
         paper_vis_config="",
         paper_final_plain=False,
         recursive_sequences=False):
    source = str(source)
    is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)
    elif not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    label_dir = save_dir / "labels"
    fourier_dir = save_dir / "fourier"
    yolo_dir = save_dir / "yolo"
    point_dir = save_dir / "point"
    point_heatmap_dir = save_dir / "point_heatmap"
    geometry_debug_dir = save_dir / "geometry_bridge"
    temporal_debug_dir = save_dir / "point_temporal"
    paper_flow_dir = save_dir / "paper_flow"
    save_dir.mkdir(parents=True, exist_ok=True)
    if not nosave:
        for output_dir in (
                fourier_dir, yolo_dir, point_dir, point_heatmap_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
        if geometry_debug:
            geometry_debug_dir.mkdir(parents=True, exist_ok=True)
    if temporal_debug:
        temporal_debug_dir.mkdir(parents=True, exist_ok=True)
    if paper_flow:
        paper_flow_dir.mkdir(parents=True, exist_ok=True)
    paper_flow_stem_filter = {
        item.strip()
        for item in str(paper_flow_frame_stems or "").split(",")
        if item.strip()
    }
    if paper_flow and paper_flow_stem_filter:
        print(
            "Paper-flow frame filter: "
            + ", ".join(sorted(paper_flow_stem_filter))
        )
    if save_txt:
        label_dir.mkdir(parents=True, exist_ok=True)

    source_root = Path(source) if Path(source).is_dir() else None
    oracle_gt = bool(oracle_gt)
    point_center_filter = bool(point_center_filter)
    point_center_radius_scale = float(point_center_radius_scale)
    point_center_min_radius = float(point_center_min_radius)
    if point_center_radius_scale <= 0.0:
        raise ValueError("--point-center-radius-scale must be greater than 0")
    if point_center_min_radius < 0.0:
        raise ValueError("--point-center-min-radius must be non-negative")
    if point_center_filter:
        print(
            "Final point-output center filter: ON "
            f"(radius=max({point_center_min_radius:g}px, "
            f"{point_center_radius_scale:g} x Fourier contour radius))")
    gt_label_root = None
    if diagnostic or oracle_gt:
        if gt_label_dir:
            gt_label_root = Path(gt_label_dir)
        elif source_root is not None and source_root.name.lower() == "images":
            gt_label_root = source_root.parent / "labels"
        else:
            gt_label_root = None
        if gt_label_root is None or not gt_label_root.is_dir():
            if oracle_gt:
                raise FileNotFoundError(
                    f"Oracle GT label directory not found: {gt_label_root}")
            print(f"WARNING: diagnostic GT label directory not found: {gt_label_root}")
            diagnostic = False
        else:
            print(f"Diagnostic GT labels: {gt_label_root}")

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
    if not getattr(model, "ft_infer", False):
        raise RuntimeError("The checkpoint does not contain an FTInfer head.")
    core_model = model.model.module if hasattr(model.model, "module") else model.model
    paper_config = _load_paper_vis_config(
        paper_vis_config) if paper_flow else None
    has_yolo_detect = (
        hasattr(core_model, "get_module_byname") and
        core_model.get_module_byname("Detect") is not None)
    if oracle_gt:
        oracle_proposals = oracle_bridges = oracle_fourier = 0
        for module in core_model.modules():
            if module.__class__.__name__ == "FourierProposal":
                module.oracle_geometry = True
                oracle_proposals += 1
            elif module.__class__.__name__ in {
                    "OutwardPointFourierFusion",
                    "CandidateDirectionalFourierBridge"}:


                module.max_gain = 0.0
                oracle_bridges += 1
            elif module.__class__.__name__ == "FTInfer":
                module.geometry_oracle = True
                oracle_fourier += 1
        if oracle_proposals == 1 and oracle_fourier == 1:
            print(
                "Strict geometry Oracle: GT macro candidate/center + "
                "GT micro directional support; predicted bridge residual off")
        else:
            print(
                "WARNING: --oracle-gt labels are available, but this "
                "checkpoint lacks the FourierProposal/FTInfer geometry path")
    print(
        "Inference candidate head: "
        f"{'YOLO Detect' if has_yolo_detect else 'FourierProposal (no Detect)'}")

    stride = model.stride
    imgsz = check_img_size(list(imgsz), s=stride)
    half = bool(half and device.type != "cpu")
    model.model.half() if half else model.model.float()

    if webcam:
        if recursive_sequences:
            raise ValueError("--recursive-sequences supports image directories only")
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        sequence_datasets = [(Path(source), dataset)]
    elif recursive_sequences:
        if source_root is None:
            raise ValueError("--recursive-sequences requires a source directory")
        sequence_dirs = _recursive_sequence_dirs(source_root)
        sequence_datasets = [
            (sequence_dir, LoadImages(
                str(sequence_dir), img_size=imgsz, stride=stride, auto=False))
            for sequence_dir in sequence_dirs
        ]
        print(
            f"Recursive sequence inference: {len(sequence_datasets)} sequences "
            f"below {source_root}")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        sequence_datasets = [(Path(source), dataset)]

    seq_len = 5
    for module in model.model.modules():
        if hasattr(module, "seq_len"):
            seq_len = int(module.seq_len)
            break
    frame_buffer = deque(maxlen=seq_len)
    oracle_path_buffer = deque(maxlen=seq_len)

    names = model.names
    ft_length = int(model.ft_length)
    ft_slice = slice(6, 6 + ft_length)
    thresholds = torch.full((len(names),), float(conf_thres), device=device)

    if device.type != "cpu":
        warmup = torch.zeros(seq_len, getattr(model.model, "ch", 3), *imgsz, device=device)
        model(warmup.type_as(next(model.model.parameters())))

    paper_hooks = None
    if paper_flow:
        paper_hooks = _PaperFeatureHooks(
            core_model, paper_config.get("feature_layers", []))

    seen = 0
    paper_saved = 0
    diagnostic_rows = []
    for (sequence_start, sequence_index, sequence_total, sequence_dir,
         current_dataset, item) in _iter_sequence_datasets(sequence_datasets):
        path, image, originals, _ = item
        if sequence_start:
            frame_buffer.clear()
            oracle_path_buffer.clear()
            try:
                sequence_name = sequence_dir.resolve().relative_to(
                    source_root.resolve()).as_posix()
            except (AttributeError, ValueError):
                sequence_name = str(sequence_dir)
            print(
                f"[sequence {sequence_index}/{sequence_total}] "
                f"{sequence_name}", flush=True)
        tensor = torch.from_numpy(image).to(device)
        tensor = tensor.half() if half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor[None]

        if not frame_buffer:
            for _ in range(seq_len - 1):
                frame_buffer.append(tensor)
                oracle_path_buffer.append(path[0] if webcam else path)
        frame_buffer.append(tensor)
        oracle_path_buffer.append(path[0] if webcam else path)
        model_input = torch.cat(list(frame_buffer), dim=0)

        if oracle_gt:
            if webcam:
                raise RuntimeError('--oracle-gt currently supports image/video files, not LoadStreams')
            GLOBAL_ORACLE_PERIPHERAL_DICT[model_input.device] = {
                'targets': _oracle_targets_for_paths(
                    list(oracle_path_buffer), gt_label_root, source_root,
                    model_input.device)
            }
        else:
            GLOBAL_ORACLE_PERIPHERAL_DICT.pop(model_input.device, None)

        predictions, point_preds = detect_ft(
            model, False, model_input, augment, thresholds, iou_thres,
            agnostic=agnostic_nms, ft_infer=True, nms_polygon=nms_polygon)
        if isinstance(predictions, list) and len(predictions) == seq_len:
            predictions = [predictions[-1]]

        for batch_index, detections in enumerate(predictions):
            source_path = path[batch_index] if webcam else path
            image_base = originals[batch_index].copy() if webcam else originals.copy()
            image_out = image_base.copy()
            yolo_out = image_base.copy()
            point_out = image_base.copy()
            stem = _name_for_frame(source_path, current_dataset)
            raw_ft_rows, box_rows, point_rows = [], [], []
            pred_polygons, pred_confidences = [], []

            if len(detections):
                detections[:, :4] = scale_coords(
                    model_input.shape[2:], detections[:, :4], image_out.shape).round()
                detections[:, ft_slice] = scale_coords_ft(
                    model_input.shape[2:], detections[:, ft_slice], image_out.shape)

                for row in detections:
                    x1, y1, x2, y2, confidence, class_id = row[:6]
                    ft_pixels = row[ft_slice].float().cpu().numpy()
                    ft_norm, polygon = _draw_raw_fourier(
                        image_out, ft_pixels, thickness=line_thickness)
                    if ft_norm is None:
                        continue

                    cls = int(class_id.item())
                    conf = float(confidence.item())
                    raw_ft_rows.append((cls, *ft_norm.tolist()))
                    pred_polygons.append(polygon)
                    pred_confidences.append(conf)
                    width, height = image_out.shape[1], image_out.shape[0]
                    if not has_yolo_detect:


                        x1 = row.new_tensor(float(polygon[:, 0].min()))
                        y1 = row.new_tensor(float(polygon[:, 1].min()))
                        x2 = row.new_tensor(float(polygon[:, 0].max()))
                        y2 = row.new_tensor(float(polygon[:, 1].max()))
                    xc = float((x1 + x2).item() * 0.5 / width)
                    yc = float((y1 + y2).item() * 0.5 / height)
                    bw = float((x2 - x1).item() / width)
                    bh = float((y2 - y1).item() / height)
                    box_rows.append((cls, xc, yc, bw, bh, conf))

                    if draw_box:
                        cv2.rectangle(image_out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                    if not diagnostic:
                        anchor = tuple(polygon[polygon[:, 1].argmin()])
                        class_name = names[cls] if not isinstance(names, dict) else names.get(cls, str(cls))
                        _draw_text(image_out, f"raw FT {class_name} {conf:.2f}",
                                   (anchor[0], anchor[1] - 4), (255, 255, 0), 0.45)

                if has_yolo_detect:

                    _draw_raw_yolo(yolo_out, detections, names, line_thickness)
                else:
                    _draw_fourier_proposals(
                        yolo_out, pred_polygons, pred_confidences,
                        line_thickness)
            else:
                if has_yolo_detect:
                    _draw_raw_yolo(yolo_out, detections, names, line_thickness)
                else:
                    _draw_fourier_proposals(
                        yolo_out, pred_polygons, pred_confidences,
                        line_thickness)

            point_batch_index = batch_index
            if isinstance(point_preds, dict) and "pred_logits" in point_preds:
                point_batch = int(point_preds["pred_logits"].shape[0])
                if len(predictions) == 1 and point_batch == seq_len:
                    point_batch_index = point_batch - 1
            point_records, point_probability = _decode_raw_point_head(
                point_preds, point_batch_index, model_input.shape[2:], image_base.shape,
                conf_threshold=point_conf_thres,
                max_points=point_max_detections)
            raw_point_count = len(point_records)
            if point_center_filter:
                point_records = _filter_points_near_fourier_centers(
                    point_records, pred_polygons,
                    radius_scale=point_center_radius_scale,
                    min_radius=point_center_min_radius)
            _draw_raw_points(
                point_out, point_records, point_conf_thres, draw_point_scores,
                raw_count=raw_point_count if point_center_filter else None)
            heatmap_out = _point_heatmap_overlay(image_base.copy(), point_probability)
            if point_probability is not None:
                cv2.rectangle(heatmap_out, (5, 5), (min(heatmap_out.shape[1] - 1, 500), 34),
                              (0, 0, 0), -1)
                cv2.putText(heatmap_out,
                            f"Raw BioCountInfer probability max={float(point_probability.max()):.4f}",
                            (11, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 1, cv2.LINE_AA)

            height, width = image_base.shape[:2]
            for x1, y1, x2, y2, score in point_records:
                xc = (x1 + x2) * 0.5 / width
                yc = (y1 + y2) * 0.5 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                point_rows.append((0, xc, yc, bw, bh, score))

            if diagnostic:
                ft_path, single_path = _resolve_label_files(
                    gt_label_root, source_path, source_root)
                if not ft_path.is_file():
                    print(f"WARNING: Fourier GT not found for {stem}: {ft_path}")
                frame_rows = _diagnostic_overlay(
                    image_out, stem, pred_polygons, pred_confidences,
                    ft_path, single_path, line_thickness, point_radius,
                    paper_style=paper_flow)
                try:
                    relative_source = source_path.resolve().relative_to(
                        source_root.resolve())
                    frame_key = relative_source.with_suffix("").as_posix()
                    sequence_value = relative_source.parent.as_posix()
                except (AttributeError, ValueError):
                    frame_key = stem
                    sequence_value = sequence_name
                for frame_row in frame_rows:
                    frame_row["sequence"] = sequence_value
                    frame_row["frame_key"] = frame_key
                diagnostic_rows.extend(frame_rows)

            if save_txt:
                with open(label_dir / f"{stem}.ft", "w", encoding="utf-8") as file:
                    for values in raw_ft_rows:
                        file.write(" ".join(f"{value:.8g}" for value in values) + "\n")
                with open(label_dir / f"{stem}.txt", "w", encoding="utf-8") as file:
                    for values in box_rows:
                        file.write(" ".join(f"{value:.8g}" for value in values) + "\n")
                with open(label_dir / f"point_{stem}.txt", "w", encoding="utf-8") as file:
                    for values in point_rows:
                        file.write(" ".join(f"{value:.8g}" for value in values) + "\n")

            if not nosave:
                _save_png(fourier_dir / f"{stem}_raw_ft.png", image_out)
                _save_png(yolo_dir / f"{stem}_raw_yolo.png", yolo_out)
                _save_png(point_dir / f"{stem}_raw_point.png", point_out)
                _save_png(point_heatmap_dir / f"{stem}_point_heatmap.png", heatmap_out)
                if geometry_debug:
                    _save_candidate_geometry_debug(
                        geometry_debug_dir, stem, image_base,
                        point_batch_index, model_input.device)
            if temporal_debug:
                _save_point_temporal_debug(
                    temporal_debug_dir, stem, model_input, core_model)
            if (paper_flow and
                    (not paper_flow_stem_filter or
                     stem in paper_flow_stem_filter) and
                    (int(paper_flow_limit) <= 0 or
                     paper_saved < int(paper_flow_limit))):
                _save_paper_flow(
                    paper_flow_dir, stem, image_base, image_out, point_out,
                    point_probability, model_input, core_model, paper_hooks,
                    point_batch_index, seq_len, paper_config,
                    plain_final=paper_final_plain)
                paper_saved += 1
            seen += 1
            if (temporal_debug and int(temporal_debug_limit) > 0 and
                    seen >= int(temporal_debug_limit)):
                break
        if (temporal_debug and int(temporal_debug_limit) > 0 and
                seen >= int(temporal_debug_limit)):
            break

    if diagnostic:
        csv_path = save_dir / "diagnostics.csv"
        fields = ["sequence", "frame_key", "frame", "prediction", "matched_gt", "confidence", "iou",
                  "gt_points", "covered_points", "point_coverage"]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        print(f"Diagnostic metrics: {csv_path}")

    if not nosave:
        print(f"Raw FT visualizations: {save_dir}")
        print(f"Raw YOLO boxes: {yolo_dir}")
        print(f"Raw point-head peaks: {point_dir}")
        print(f"Raw point-head heatmaps: {point_heatmap_dir}")
    if geometry_debug and not nosave:
        print(f"Candidate geometry diagnostics: {geometry_debug_dir}")
    if temporal_debug:
        print(f"Point temporal diagnostics: {temporal_debug_dir}")
    if paper_flow:
        print(f"Paper flow visualizations: {paper_flow_dir} ({paper_saved} frames)")
    if paper_hooks is not None:
        paper_hooks.remove()
    print(f"Processed frames: {seen}")
    return save_dir


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default="weights/best.pt",
                        help="RepViT-M0.9 checkpoint")


    parser.add_argument("--source", type=str, default="data/example/images")
    parser.add_argument(
        "--recursive-sequences", action="store_true",
        help=("Recursively discover image-sequence subdirectories below --source, "
              "load the model once, and reset the temporal buffer per sequence."))


    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640])
    parser.add_argument("--conf-thres", type=float, default=0.30,
                        help="Fourier candidate confidence threshold (historical sweep: 0.20).")
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="0")

    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--nms-polygon", type=int, default=1)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--draw-box", action="store_true")
    parser.add_argument("--no-save-txt", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--gt-label-dir", default="",
                        help="GT label root containing <stem>.ft and single_<stem>.txt; auto-detected beside images.")
    parser.add_argument("--no-diagnostic", action="store_true",
                        help="Disable GT, IoU, and single-target coverage overlays.")
    parser.add_argument("--oracle-gt", action="store_true",
                        help="Feed YOLO-box GT and single-target GT to the Oracle Fourier bridge.")
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument("--point-conf-thres", type=float, default=0.60,
                        help="Raw BioCountInfer threshold used by the historical sweep.")
    parser.add_argument("--point-max-detections", type=int, default=100,
                        help="Safety cap for raw point peaks; no spatial aggregation is applied.")
    parser.add_argument("--draw-point-scores", action="store_true")
    parser.add_argument(
        "--no-point-center-filter", action="store_true",
        help=("Disable the inference-only final point filter. By default, only "
              "points near a retained Fourier candidate center are exported/drawn."))
    parser.add_argument(
        "--point-center-radius-scale", type=float, default=1.75,
        help=("Point acceptance radius as a multiple of each predicted Fourier "
              "contour's maximum center-to-boundary radius."))
    parser.add_argument(
        "--point-center-min-radius", type=float, default=20.0,
        help="Minimum point acceptance radius in output-image pixels.")
    parser.add_argument("--no-geometry-debug", action="store_true",
                        help="Disable candidate/support/bridge feature maps and directional-radius exports.")
    parser.add_argument("--temporal-debug", action="store_true",
                        help="Export raw differences, aligned feature residuals, motion evidence, and gates.")
    parser.add_argument("--temporal-debug-limit", type=int, default=0,
                        help="Stop after N frames in temporal-debug mode; 0 processes the full source.")
    parser.add_argument(
        "--paper-flow", action="store_true",
        help=("Export a curated input/backbone/point/Fourier/final-contour "
              "figure plus exact feature maps; inference only."))
    parser.add_argument(
        "--paper-flow-limit", type=int, default=0,
        help="Save paper-flow outputs for at most N frames; 0 saves all frames.")
    parser.add_argument(
        "--paper-flow-frame-stems", type=str, default="",
        help=("Optional comma-separated image stems to export in --paper-flow "
              "mode. All frames are still inferred so temporal state remains valid."))
    parser.add_argument(
        "--paper-vis-config", type=str,
        default=str(ROOT / "configs" / "paper_visualization_t2.json"),
        help=("JSON settings for hooked layers, labels, color scales and "
              "paper-flow layout."))
    parser.add_argument(
        "--paper-final-plain", action="store_true",
        help=("Save the final paper-flow contour as a plain image with only "
              "large cluster identifiers; omit the title bar and metric HUD."))
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument(
        "--csv-only", action="store_true",
        help=("Diagnostic batch mode: save only diagnostics.csv; disables "
              "rendered images, prediction txt files, geometry maps, temporal "
              "debug maps, and paper-flow exports."))
    opt = parser.parse_args()
    opt.half = not opt.no_half
    opt.save_txt = not opt.no_save_txt
    opt.diagnostic = not opt.no_diagnostic
    opt.geometry_debug = not opt.no_geometry_debug
    opt.point_center_filter = not opt.no_point_center_filter
    if opt.csv_only:
        opt.diagnostic = True
        opt.nosave = True
        opt.save_txt = False
        opt.geometry_debug = False
        opt.temporal_debug = False
        opt.paper_flow = False
    del (opt.no_half, opt.no_save_txt, opt.no_diagnostic,
         opt.no_geometry_debug, opt.no_point_center_filter, opt.csv_only)


    return opt


if __name__ == "__main__":
    run(**vars(parse_opt()))
