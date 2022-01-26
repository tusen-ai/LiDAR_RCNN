from __future__ import print_function
import argparse
import os
import random
import pprint
import timeit
import yaml
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data

import torch.nn.functional as F
from easydict import EasyDict as edict

from LiDAR_RCNN.core.function import train
from LiDAR_RCNN.models.point_net import PointNet
from LiDAR_RCNN.utils.model_utils import FullModel
from LiDAR_RCNN.utils.utils import create_logger, get_world_size

from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                    help='experiment cfgure file name',
                    required=True,
                    type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--name", type=str, default="")

args = parser.parse_args()
cfg = edict(yaml.load(open(args.cfg, 'r')))
logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')
logger.info(pprint.pformat(args))
logger.info(cfg)

writer_dict = {
'writer': SummaryWriter(tb_log_dir),
'train_global_steps': 0,
'valid_global_steps': 0,
}

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
cudnn.enabled = cfg.CUDNN.ENABLED
distributed = cfg.nGPUS > 1
device = torch.device('cuda:{}'.format(args.local_rank))

if distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
    )

BaseDataset = importlib.import_module("LiDAR_RCNN.datasets." + cfg.DATASET).TFRecordDataset

tfrecord_path = os.path.join(cfg.TRAIN.DATA_PATH, 'train.rec')
index_path = os.path.join(cfg.TRAIN.DATA_PATH, 'train.idx')
description = {"name": "byte", "data": "byte"}
train_dataset = BaseDataset(get_world_size(), tfrecord_path, index_path, description, points_num=cfg.TRAIN.NUM_POINTS, frame=cfg.MODEL.Frame, shuffle_queue_size=cfg.TRAIN.SUFFLE_SIZE, rank=args.local_rank, train=True, iou_threshold=cfg.TRAIN.IOU_THRESHOLD, valid_cls=cfg.TRAIN.VALID_CLS)

trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=None)

model = eval(cfg.MODEL.NAME)(cfg.MODEL.PTS_DIM, cfg.MODEL.X, cfg.MODEL.CLS_NUM)
model.init_weights()
model = FullModel(model, cfg)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = model.to(device)
model = nn.parallel.DistributedDataParallel(
model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

# optimizer
if cfg.TRAIN.OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD([{'params':
                                filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                'lr': cfg.TRAIN.LR}],
                            lr=cfg.TRAIN.LR,
                            momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WD,
                            nesterov=cfg.TRAIN.NESTEROV,
                            )
elif cfg.TRAIN.OPTIMIZER == 'adamw':
    optimizer = torch.optim.AdamW([{'params':
                                filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                'lr': cfg.TRAIN.LR}],
                            lr=cfg.TRAIN.LR,
                            weight_decay=cfg.TRAIN.WD,
                            )

with open(index_path, 'r') as f:
    DATA_LEN = len(f.readlines())
epoch_iters = np.int(DATA_LEN /
                cfg.TRAIN.BATCH_SIZE_PER_GPU / cfg.nGPUS)

scheduler  = torch.optim.lr_scheduler.OneCycleLR(
                                optimizer,
                                max_lr = cfg.TRAIN.LR,
                                steps_per_epoch = epoch_iters,
                                epochs = cfg.TRAIN.END_EPOCH,
                            )

start = timeit.default_timer()
end_epoch = cfg.TRAIN.END_EPOCH
num_iters = cfg.TRAIN.END_EPOCH * epoch_iters
last_epoch = 0
if cfg.TRAIN.RESUME:
    model_state_file = os.path.join(cfg.TRAIN.MODEL_PATH)
    if os.path.isfile(model_state_file):
        checkpoint = torch.load(model_state_file,
                map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['epoch']
        logger.info("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

if cfg.TRAIN.PRETRAIN:
    model_state_file = os.path.join(cfg.TRAIN.MODEL_PATH)
    if os.path.isfile(model_state_file):
        checkpoint = torch.load(model_state_file,
                map_location=lambda storage, loc: storage)
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))


for epoch in range(last_epoch, end_epoch):
    train(cfg, epoch, cfg.TRAIN.END_EPOCH,
            epoch_iters, cfg.TRAIN.LR, num_iters,
            trainloader, optimizer, scheduler, model, writer_dict, device, final_output_dir)
    if args.local_rank == 0 and epoch % 2 != 0:
        logger.info('=> saving checkpoint to {}'.format(
                os.path.join(final_output_dir,'checkpoint_{}_{}.pth.tar'.format(args.name, epoch))))
        torch.save({
            'epoch': epoch+1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint_{}_{}.pth.tar'.format(args.name, epoch)))

writer_dict['writer'].close()
end = timeit.default_timer()
logger.info('Hours: %d' % np.int((end-start)/3600))
logger.info('Done')
