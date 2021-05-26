import torch
import typing
import random
import numpy as np
import pickle as pkl
from tfrecord import reader
from tfrecord import iterator_utils
from LiDAR_RCNN.datasets.waymo.data_utils import *


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 world_size: int,
                 data_path: str,
                 index_path: typing.Union[str, None],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 points_num: typing.Optional[int] = 512,
                 iou_threshold: typing.Optional[typing.List[str]] = [0.7],
                 valid_cls: typing.Optional[typing.List[int]] = [0, 1],
                 shuffle_queue_size: typing.Optional[int] = None,
                 rank: typing.Optional[int] = None,
                 train: typing.Optional[bool] = False,
                 transform: typing.Callable[[dict], typing.Any] = None
                 ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size
        self.points_num = points_num
        self.rank = rank
        self.train = train
        self.world_size = world_size
        self.iou_threshold = iou_threshold
        self.valid_cls = valid_cls
        self.transform = self.transform_train if self.train else self.transform_test

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard = self.rank * worker_info.num_workers + worker_info.id, worker_info.num_workers * self.world_size
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = reader.tfrecord_loader(
            self.data_path, self.index_path, self.description, shard)
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it

    def transform_train(self, it):
        pcd, proposal, gt_box, gt_cls = load_data(it)
        if gt_cls == 1:
            # only use jitter for vechile
            proposal = jitter(proposal, 0.5)

        gt_cls = relabel_by_iou(proposal, gt_box, gt_cls, self.iou_threshold)
        # set proposals without points as nagtive
        if pcd.shape[0] == 0:
            gt_cls = 0  
        point_set = process_pcd(pcd, proposal, self.points_num)

        # heading residual
        gt_box[-1] = get_heading_residual(gt_box[-1], proposal[-1])

        # disable other cls
        if gt_cls not in self.valid_cls:
            gt_cls = 0

        # move gt box to pred center
        gt_box[:3] -= proposal[:3]
        gt_box[:2] = rotz(-proposal[-1]) @ gt_box[:2]
        return point_set.astype(np.float32), proposal.astype(np.float32), gt_cls, gt_box.astype(np.float32)

    def transform_test(self, it):
        name = "".join([chr(item) for item in it['name']])
        pcd, proposal, gt_box, gt_cls = load_data(it)
        point_set = process_pcd(pcd, proposal, self.points_num)
        return point_set.astype(np.float32), proposal.astype(np.float32), name

    