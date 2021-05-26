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
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter
from collections import defaultdict
from easydict import EasyDict as edict

from LiDAR_RCNN.core.function import test
from LiDAR_RCNN.models.point_net import PointNet
from LiDAR_RCNN.utils.model_utils import FullModel
from LiDAR_RCNN.utils.utils import get_world_size

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                    help='experiment cfgure file name',
                    required=True,
                    type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default='')

args = parser.parse_args()
cfg = edict(yaml.load(open(args.cfg, 'r')))

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
cudnn.enabled = cfg.CUDNN.ENABLED
distributed = cfg.nGPUS > 1
device = torch.device('cuda:{}'.format(args.local_rank))
if distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
    )

model = eval(cfg.MODEL.NAME)(cfg.MODEL.PTS_DIM, cfg.MODEL.X, cfg.MODEL.CLS_NUM)
# load model

pretrained_dict = torch.load(args.checkpoint,
                             map_location=torch.device('cpu'))['state_dict']
model_dict = model.state_dict()
pretrained_dict = {
    k[6:]: v
    for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()
}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

BaseDataset = importlib.import_module("LiDAR_RCNN.datasets." + cfg.DATASET).TFRecordDataset
tfrecord_path = os.path.join(cfg.TRAIN.DATA_PATH, 'val.rec')
index_path = os.path.join(cfg.TRAIN.DATA_PATH, 'val.idx')
description = {"name": "byte", "data": "byte"}
val_dataset = BaseDataset(get_world_size(),
                          tfrecord_path,
                          index_path,
                          description,
                          points_num=cfg.TEST.NUM_POINTS,
                          rank=args.local_rank,
                          valid_cls=cfg.TRAIN.VALID_CLS,
                          train=False)
valloader = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
                                        shuffle=False,
                                        num_workers=cfg.TEST.WORKERS,
                                        pin_memory=True,
                                        drop_last=False,
                                        sampler=None)

model = model.to(device)
model = nn.parallel.DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank)
test(cfg, 0, valloader, model, device, cfg.TEST.TAT_PATH)
