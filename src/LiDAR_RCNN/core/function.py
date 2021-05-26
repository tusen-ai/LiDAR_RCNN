import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import pickle
import torch.nn as nn
from torch.nn import functional as F

import torch.distributed as dist
from LiDAR_RCNN.utils.utils import *
from LiDAR_RCNN.utils.model_utils import from_prediction_to_label_format
# from datasets.tusimple.dataset_tfrecord import data_prefetcher
# from LiDAR_RCNN.utils.bbox_utils import get_3d_box, box3d_iou, get_2d_bbox, get_2d_iou


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(cfg, epoch, num_epoch, epoch_iters, base_lr, num_iters, trainloader,
          optimizer, model, writer_dict, device, final_output_dir):
    # Training
    model.train()
    batch_time = 0
    ave_loss = AverageMeter()
    ave_cls_loss = AverageMeter()
    ave_center_loss = AverageMeter()
    ave_size_loss = AverageMeter()
    ave_heading_loss = AverageMeter()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    rank = get_rank()
    world_size = get_world_size()

    tic = time.time()
    for i_iter, batch in enumerate(trainloader):
        pts, pred_bbox, cls_labels, reg_labels = batch
        cls_labels = cls_labels.float().to(device)
        reg_labels = reg_labels.float().to(device)
        pred_bbox = pred_bbox.float().to(device)
        pts = pts.float().to(device)
        loss, cls_loss, center_loss, size_loss, heading_loss = model(
            pts, pred_bbox, cls_labels, reg_labels)

        reduced_loss = reduce_tensor(loss)
        reduced_cls_loss = reduce_tensor(cls_loss)
        reduced_center_loss = reduce_tensor(center_loss)
        reduced_size_loss = reduce_tensor(size_loss)
        reduced_heading_loss = reduce_tensor(heading_loss)

        ave_loss.update(reduced_loss.item())
        ave_cls_loss.update(reduced_cls_loss.item())
        ave_center_loss.update(reduced_center_loss.item())
        ave_size_loss.update(reduced_size_loss.item())
        ave_heading_loss.update(reduced_heading_loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time += time.time() - tic
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer, base_lr, num_iters,
                                  i_iter + cur_iters)
        if i_iter % 5000 == 0:
            torch.cuda.empty_cache()

        if i_iter % cfg.PRINT_FREQ == 0 and rank == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}, cls_loss: {:.6f}, center_loss: {:.6f}, size_loss: {:.6f},   heading_loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time / cfg.PRINT_FREQ, lr, ave_loss.average() / world_size, ave_cls_loss.average() / world_size, ave_center_loss.average() / world_size, ave_size_loss.average() / world_size, ave_heading_loss.average() / world_size)
            logging.info(msg)
            batch_time = 0
    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def test(cfg, epoch, testloader, model, device, target_path):
    model.eval()
    rank = get_rank()
    results = []
    scores = []
    names = []
    preds = []
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            if rank == 0 and idx % 100 == 0:
                print(idx)
            pts, pred_bbox, name = batch
            pred_bbox = pred_bbox.float().to(device)
            pts = pts.float().to(device)

            # a = time.time()
            logits, centers, sizes, headings = model(pts, pred_bbox)
            logits = logits.cpu().numpy()
            centers = centers.cpu().numpy()

            sizes = sizes.cpu().numpy()
            headings = headings.cpu().numpy()
            pred_bbox = pred_bbox.cpu().numpy()

            l, w, h, tx, ty, tz, ry = from_prediction_to_label_format(
                centers, sizes, headings, pred_bbox)
            csr = np.vstack([l, w, h, tx, ty, tz, ry]).T
            results.append(csr)
            scores.append(logits)
            names.append(name)
            preds.append(pred_bbox)
            torch.cuda.synchronize()
        results = np.vstack(results)
        scores = np.vstack(scores)
        preds = np.vstack(preds)
        output_name = os.path.join(target_path, "results_{}.pkl".format(rank))
        with open(output_name, 'wb') as f:
            pickle.dump(results, f)
            pickle.dump(scores, f)
            pickle.dump(preds, f)
            pickle.dump(names, f)
