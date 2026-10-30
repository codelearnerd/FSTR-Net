import numpy as np

def polygon_area_batch(polys, mask=None):


    if mask is not None:


        first_points = polys[:, :1, :]
        polys = np.where(mask[..., None], polys, first_points)
    x = polys[..., 0]
    y = polys[..., 1]
    return 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    )

def clip_to_rect_batch(polys, xmin, ymin, xmax, ymax):


    nt, npts, _ = polys.shape

    clipped = polys.copy()
    clipped[..., 0] = np.clip(clipped[..., 0], xmin, xmax)
    clipped[..., 1] = np.clip(clipped[..., 1], ymin, ymax)


    mask = np.ones((nt, npts), dtype=bool)
    return clipped, mask

def out_range_filt_poly_batch(poly_ptses, shape, iou_thresh=0.2):


    if len(shape) == 2:
        xmin, ymin, xmax, ymax = 0, 0, shape[1], shape[0]
    else:
        xmin, ymin, xmax, ymax = shape

    nt, _ = poly_ptses.shape
    npts = poly_ptses.shape[1] // 2
    polys = poly_ptses.reshape(nt, npts, 2)


    obj_area = polygon_area_batch(polys)


    inter_poly, mask = clip_to_rect_batch(polys, xmin, ymin, xmax, ymax)
    inter_area = polygon_area_batch(inter_poly, mask)


    ioa = np.zeros(nt, dtype=np.float32)
    valid = obj_area > 1e-9
    ioa[valid] = inter_area[valid] / obj_area[valid]

    polys_out_ids = np.where(ioa > iou_thresh)[0]
    return polys_out_ids, ioa
