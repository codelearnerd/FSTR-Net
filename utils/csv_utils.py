import cv2
import numpy as np
import torch
import math
import csv

def ais_to_pixel(x, y, A23):
    original_vector = np.array([x, y, 1])
    result_vector = A23 @ original_vector

    x_prime = result_vector[0]
    y_prime = result_vector[1]


    return int(round(x_prime)), int(round(y_prime))

def read_csv_points(path, columns=("lon", "lat", "altitude", "speed")):


    ais = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                values = []
                for col in columns:
                    values.append(float(r.get(col, 0.0)))
                ais.append(values)
            except Exception:
                continue

    return np.array(ais, dtype=float) if ais else np.zeros((0, len(columns)))

def csv_to_img(csv_path, scale_factor = 500, max_allowed = 300000000):
    ais = read_csv_points(csv_path)
    if ais.shape[0] > 0:

        aisxy = ais[:,:2]

        center = aisxy.mean(axis=0)
        pts = (aisxy - center) * scale_factor


        eps = 1e-5
        h = ais[:,-1]
        h = (h-h.min())/(h.max()-h.min()+eps)

        v = ais[:,-2]
        v = (v-v.min())/(v.max()-v.min()+eps)


        min_x, min_y = pts[:,0].min(), pts[:,1].min()
        max_x, max_y = pts[:,0].max(), pts[:,1].max()
        inner_w = max_x - min_x
        inner_h = max_y - min_y
        if inner_w == 0: inner_w = 1.0
        if inner_h == 0: inner_h = 1.0

        margin = 50
        img_w = int(math.ceil(inner_w + 2*margin))
        img_h = int(math.ceil(inner_h + 2*margin))


        scale = 1.0
        if inner_w + 2*margin > max_allowed or inner_h + 2*margin > max_allowed:
            scale = min((max_allowed - 2*margin)/inner_w,
                        (max_allowed - 2*margin)/inner_h)
            img_w = int(math.ceil(inner_w*scale + 2*margin))
            img_h = int(math.ceil(inner_h*scale + 2*margin))


        k, offset = scale*scale_factor, margin - scale*(center*scale_factor + np.array([min_x,min_y]))
        A23 = np.array([[k, 0, offset[0]],
                        [0, k, offset[1]]],dtype=np.float32)

        puv = [ais_to_pixel(x,y,A23) for x,y in aisxy]
        assert len(puv)==h.shape[0] and len(puv)==v.shape[0]
        puv = np.array(puv)

        img = np.zeros((img_h, img_w, 3), dtype=np.uint8) * 255


        for i in range(len(puv) - 1):
            p1 = tuple(puv[i])
            p2 = tuple(puv[i + 1])

            color = (
                int(v[i] * 255),
                int(h[i] * 255),
                255
            )
            cv2.line(img, p1, p2, color=color, thickness=3, lineType=cv2.LINE_AA)
    else:
        img = None
        puv = np.empty((0, 2))
        A23 = None
    return img,ais,puv,A23


def find_points_in_quadrilateral_simple(track, obj_poly,tolance=5.0):


    poly = obj_poly.astype(np.int32).reshape(-1, 1, 2)

    subid = []
    for i, point in enumerate(track):

        point_int = (int(point[0]), int(point[1]))


        dist = cv2.pointPolygonTest(poly, point_int, True)
        if dist >= -tolance:
            subid.append(i)

    return torch.tensor(subid), torch.from_numpy(track[subid])

def generate_convex_hull(points):

    if len(points) < 3:
        return None


    points_float32 = points.astype(np.float32)
    hull = cv2.convexHull(points_float32)
    return hull.reshape(-1, 2)


def find_continuous_segments_v2(ids,keeps):

    if len(ids) == 0:
        return torch.empty((0, 3), dtype=torch.long)


    valid_mask = ids >= 0
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return torch.empty((0, 3), dtype=torch.long)


    valid_values = ids[valid_indices]


    changes = torch.zeros(len(valid_indices) + 1, dtype=torch.bool, device=ids.device)
    changes[0] = True
    changes[1:-1] = (valid_values[1:] != valid_values[:-1]) | (valid_indices[1:] != valid_indices[:-1] + 1)
    changes[-1] = True

    change_positions = torch.where(changes)[0]

    segments = []
    for i in range(len(change_positions) - 1):
        start = change_positions[i]
        end = change_positions[i + 1] - 1
        kid = valid_values[start]
        start_idx = valid_indices[start]
        end_idx = valid_indices[end]
        obj = keeps[kid]
        segments.append([kid, start_idx, end_idx, obj[-1],obj[-2]])

    return torch.tensor(segments, dtype=torch.float32)


def find_continuous_segments_v4(ids, keeps, max_gap=6):

    if len(ids) == 0:
        return torch.empty((0, 5), dtype=torch.float32)

    valid_mask = ids >= 0
    valid_indices = torch.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return torch.empty((0, 5), dtype=torch.float32)

    valid_values = ids[valid_indices]


    changes = torch.zeros(len(valid_indices) + 1, dtype=torch.bool, device=ids.device)
    changes[0] = True
    changes[1:-1] = (valid_values[1:] != valid_values[:-1]) | (valid_indices[1:] != valid_indices[:-1] + 1)
    changes[-1] = True

    change_positions = torch.where(changes)[0]


    seg_kids = []
    seg_starts = []
    seg_ends = []
    seg_cls = []
    seg_conf = []

    for i in range(len(change_positions) - 1):
        start_pos = change_positions[i]
        end_pos = change_positions[i + 1] - 1
        kid = valid_values[start_pos]
        start_idx = valid_indices[start_pos]
        end_idx = valid_indices[end_pos]
        obj = keeps[kid]

        seg_kids.append(kid)
        seg_starts.append(start_idx)
        seg_ends.append(end_idx)
        seg_cls.append(obj[-1])
        seg_conf.append(obj[-2])


    merged_segments = []
    n_segments = len(seg_kids)

    if n_segments == 0:
        return torch.empty((0, 5), dtype=torch.float32)

    current_kid = seg_kids[0]
    current_start = seg_starts[0]
    current_end = seg_ends[0]
    current_cls = seg_cls[0]
    current_conf = seg_conf[0]

    for i in range(1, n_segments):
        if (seg_kids[i] == current_kid and
            seg_starts[i] - current_end - 1 <= max_gap):

            current_end = seg_ends[i]
        else:

            merged_segments.append([current_kid, current_start, current_end, current_cls, current_conf])

            current_kid = seg_kids[i]
            current_start = seg_starts[i]
            current_end = seg_ends[i]
            current_cls = seg_cls[i]
            current_conf = seg_conf[i]


    merged_segments.append([current_kid, current_start, current_end, current_cls, current_conf])

    return torch.tensor(merged_segments, dtype=torch.float32)

