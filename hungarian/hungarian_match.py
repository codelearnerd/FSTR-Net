
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from hungarian.ious import bbox_overlaps

def bbox_iou(boxA, boxB):
    """
    Compute the Intersection Over Union value between two bounding boxes
    Each box is defined by its center, width and height: [x_center, y_center, width, height]
    """


    boxA = [boxA[0] - boxA[2] / 2.0, boxA[1] - boxA[3] / 2.0, boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0]
    boxB = [boxB[0] - boxB[2] / 2.0, boxB[1] - boxB[3] / 2.0, boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0]


    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)


    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)


    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def hungarian_match(predicted_boxes, true_boxes):
    ntruth = true_boxes.size(0)
    npred = predicted_boxes.size(0)


    ious = bbox_overlaps(true_boxes,predicted_boxes, mode='iou')
    '''
    ious = torch.zeros(ntruth, npred)
    for i in range(ntruth):
        for j in range(npred):
            ious[i, j] = bbox_iou(true_boxes[i], predicted_boxes[j])
    '''

    true_centers = true_boxes[:, :2].unsqueeze(1)
    pred_centers = predicted_boxes[:, :2].unsqueeze(0)
    center_diffs = (true_centers - pred_centers).pow(2).sum(dim=-1).sqrt()
    center_diffs /= center_diffs.max()


    modified_ious = center_diffs - ious


    row_idx, col_idx = linear_sum_assignment(modified_ious.detach().cpu().numpy())
    assert(row_idx.shape[0] == col_idx.shape[0])
    n_matched = min(ntruth,npred)
    assert(row_idx.shape[0]==n_matched)
    matched_ious = ious[row_idx, col_idx]
    assert(matched_ious.shape[0]==n_matched)

    return ious,matched_ious,(row_idx, col_idx)

def hungarian_loss(predicted_boxes, true_boxes):
    ious,matched_ious,(row_idx, col_idx) = hungarian_match(predicted_boxes, true_boxes)


    loss = (1 - matched_ious).sum()

    column_selected = torch.zeros(ious.shape[1], dtype=torch.float32)
    column_selected[col_idx] = 1

    return loss,column_selected,(row_idx, col_idx)


