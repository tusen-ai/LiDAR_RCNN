import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple


class FullModel(nn.Module):
    def __init__(self, model, cfg):
        super(FullModel, self).__init__()
        self.model = model
        self.loss_weight = cfg.TRAIN.LOSS_WEIGHT
        self.loss_center = nn.SmoothL1Loss(reduction='none')
        self.loss_size = nn.SmoothL1Loss(reduction='none')
        self.loss_heading = nn.SmoothL1Loss(reduction='none')

    def forward(self, inputs, pred_bbox, cls_labels, reg_labels):
        logits, centers, sizes, headings = self.model(inputs, pred_bbox)
        loss, cls_loss, center_loss, size_loss, heading_loss = self.get_loss(
            logits, centers, sizes, headings, cls_labels, reg_labels,
            pred_bbox)
        return loss, cls_loss, center_loss, size_loss, heading_loss

    def get_loss(self, logits, centers, sizes, headings, cls_labels,
                 reg_labels, pred_bbox):
        center_labels, residual_size_labels, residual_angle_labels = self.parse_labels(
            cls_labels, reg_labels, pred_bbox)

        reg_loss_valid_mask = (cls_labels > 0).long()
        cls_loss = nn.CrossEntropyLoss()(logits, cls_labels.long()).mean()

        center_loss = (self.loss_center(centers,
                                       center_labels).mean(dim=-1) * reg_loss_valid_mask).sum() / (
                                           reg_loss_valid_mask.sum() + 1e-5)
        size_loss = (self.loss_size(
            sizes, residual_size_labels).mean(dim=-1) * reg_loss_valid_mask).sum() / (
                reg_loss_valid_mask.sum() + 1e-5)
        heading_loss = (self.loss_heading(
            headings.reshape(-1), residual_angle_labels) * reg_loss_valid_mask).sum() / (
                reg_loss_valid_mask.sum() + 1e-5)

        loss = cls_loss + self.loss_weight * (center_loss + size_loss +
                                              heading_loss)

        return loss, cls_loss, center_loss, size_loss, heading_loss

    def parse_labels(self, cls_labels, reg_labels, pred_bbox):
        centers = reg_labels[:, [0, 1, 2]] / pred_bbox[:, [3, 4, 5]]
        residual_sizes = torch.log(reg_labels[:, [3, 4, 5]] /
                                   pred_bbox[:, [3, 4, 5]])
        return centers, residual_sizes, reg_labels[:, -1]

def from_prediction_to_label_format(centers, sizes, headings,
                                    pred_bbox):
    l, w, h = (np.exp(sizes) * pred_bbox[:, [3, 4, 5]]).T
    ry = headings.reshape(-1)
    tx, ty, tz = (centers * pred_bbox[:, [3, 4, 5]]).T
    return l, w, h, tx, ty, tz, ry

def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                    [s,  c]])

def back_to_lidar_coords(f_bbox, pred_bbox):
    f_bbox[-1] = f_bbox[-1] + pred_bbox[6]
    f_bbox[:2] = rotz(pred_bbox[6]) @ f_bbox[:2]
    f_bbox[:3] += pred_bbox[:3]
    return f_bbox