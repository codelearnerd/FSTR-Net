"""Finite polygon construction for Fourier metrics and the GEOS NMS fallback."""

from collections import Counter
import warnings

import numpy as np
from shapely.errors import ShapelyError
from shapely.geometry import Polygon


DIAGNOSTICS = Counter()


def record_invalid_geometry(reason, count=1):
    first = DIAGNOSTICS[reason] == 0
    DIAGNOSTICS[reason] += int(count)
    if first:
        warnings.warn(
            f'Invalid geometry: {reason} ({count} occurrence(s)). '
            'Invalid polygons have zero IoU; non-finite predictions are '
            'excluded from NMS. Check model outputs and labels if this recurs.',
            RuntimeWarning, stacklevel=2)


def safe_polygon(points):
    points = np.asarray(points, dtype=np.float64)
    if points.size % 2:
        raise ValueError('Polygon coordinates must contain x/y pairs')
    points = points.reshape(-1, 2)
    if not np.isfinite(points).all():
        record_invalid_geometry('nonfinite_polygon_coordinates')
        return Polygon()
    if len(points) < 3 or len(np.unique(points, axis=0)) < 3:
        record_invalid_geometry('degenerate_polygon')
        return Polygon()
    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.is_valid or not np.isfinite(polygon.area) or polygon.area <= 0:
            record_invalid_geometry('degenerate_polygon')
            return Polygon()
        return polygon
    except ShapelyError:
        record_invalid_geometry('geos_polygon_construction_failed')
        return Polygon()


def safe_polygon_iou(first, second):
    if first.is_empty or second.is_empty:
        return 0.0
    try:
        intersection = first.intersection(second).area
        union = first.area + second.area - intersection
        if not np.isfinite(intersection) or not np.isfinite(union):
            record_invalid_geometry('nonfinite_polygon_area')
            return 0.0
        return float(np.clip(intersection / union, 0, 1)) if union > 0 else 0.0
    except ShapelyError:
        record_invalid_geometry('geos_polygon_intersection_failed')
        return 0.0
