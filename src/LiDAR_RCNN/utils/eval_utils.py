import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from torch.nn import functional as F
from multiprocessing import Process, Queue

from LiDAR_RCNN.utils.nms import wnms_wrapper
from LiDAR_RCNN.utils.model_utils import back_to_lidar_coords

from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.protos import metrics_pb2


def merge_results(target_path, nGPUS):
    data_lst = []
    socre_lst = []
    name_lst = []
    preds_lst = []
    for i in range(nGPUS):
        with open(os.path.join(target_path, "results_{}.pkl".format(i)), 'rb') as f:
            data = pkl.load(f)
            socres = pkl.load(f)
            preds = pkl.load(f)
            names = pkl.load(f)
        data_lst.append(data)
        socre_lst.append(socres)
        preds_lst.append(preds)
        for name in names:
            name_lst += name
    data_lst = np.vstack(data_lst)[:, [3,4,5,0,1,2,6]]
    socre_lst = np.vstack(socre_lst)
    preds_lst = np.vstack(preds_lst)

    outputs_bboxes = defaultdict(list)
    outputs_socres = defaultdict(list)
    for i in tqdm(range(len(name_lst))):
        f_bbox = back_to_lidar_coords(data_lst[i].copy(), preds_lst[i])
        name = '/'.join(name_lst[i].split('/')[:2])
        outputs_bboxes[name].append(f_bbox)
        outputs_socres[name].append(socre_lst[i])
    return outputs_bboxes, outputs_socres


def do_nms(output_dict, scores_dict):
    data_queue = Queue()
    result_queue = Queue()
    workers = []
    num_workers = 10

    def eval_worker(data_queue, result_queue):
        while True:
            k, cid, data = data_queue.get()
            nms = wnms_wrapper(0.1, 0.5)
            # nms = wnms_wrapper(0.1, 0.999)
            det = nms(data)
            result_queue.put((k, cid, det))
            # result_queue.put((k, cid, data))
    for _ in range(num_workers):
        workers.append(Process(target=eval_worker, args=(data_queue, result_queue)))
    for w in workers:
        w.daemon = True
        w.start()

    for k in tqdm(output_dict):
        bbox_csr = np.array(output_dict[k])
        logits = np.array(scores_dict[k])
        cls_score = F.softmax(torch.from_numpy(logits / 4), dim=-1).numpy()
        box_corners = box_utils.get_upright_3d_box_corners(bbox_csr).numpy()
        box_corners_2d = box_corners[:, :4, :2].reshape(-1, 8)
        heading = bbox_csr[:, -1].reshape(-1, 1)
        z0 = box_corners[:, 0, 2].reshape(-1, 1)
        logh = np.log(bbox_csr[:, 5]).reshape(-1, 1)
        cls_num = cls_score.shape[-1]
        for cid in range(1, cls_num):
            score = cls_score[:, cid]
            # valid_inds = score > 0.1
            det = np.concatenate((box_corners_2d, heading, z0, logh, score.reshape(-1,1)), axis=1)
            # det = det[valid_inds]
            data_queue.put((k, cid, det))

    final_dets_dict = defaultdict(dict)
    for i in tqdm(output_dict):
        for j in range(1, cls_num):
            k, cid, det = result_queue.get()
            final_dets_dict[k][cid] = det
    return final_dets_dict


def _create_bbox_prediction(det, class_id, frame_name, marco_ts):
    o = metrics_pb2.Object()
    o.context_name = (frame_name)
    o.frame_timestamp_micros = marco_ts
    box = label_pb2.Label.Box()
    box.center_x = np.mean(det[[0,2,4,6]])
    box.center_y = np.mean(det[[1,3,5,7]])
    z0 = det[9]
    height = np.exp(det[10])
    box.center_z = z0 + height / 2
    box.width = np.sqrt((det[2] - det[4]) ** 2 + (det[3] - det[5]) ** 2)
    box.length = np.sqrt((det[2] - det[0]) ** 2 + (det[3] - det[1]) ** 2)
    box.height = height
    box.heading = det[8]
    o.object.box.CopyFrom(box)
    if len(det) == 12:
        o.score = det[11]
    o.object.id = ''
    o.object.type = class_id
    return o

def _create_pd_file(obj_list, filename):
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()
    for obj in obj_list:
        objects.objects.append(obj)
    f = open(filename, 'wb')
    f.write(objects.SerializeToString())
    f.close()


def create_bin(output_dict, target_path, name='tusimple'):
    pred_list = []
    count = 0
    rec_id_to_ts = {}
    rec_id_to_frame_name = {}
    for rec_id, v in output_dict.items():
        count += 1
        if count % 1000 == 0:
            print(count, '/', len(output_dict))
        frame_name = rec_id.split('/')[0]
        ts = int(rec_id.split('/')[1])
        rec_id_to_frame_name[rec_id] = frame_name
        rec_id_to_ts[rec_id] = ts
        for i in range(1, 5): # 5 is for waymo class number
            if i in v.keys():
                for bbox in v[i]:
                    pred_list.append(_create_bbox_prediction(bbox, i, frame_name, ts))
    _create_pd_file(pred_list, os.path.join(target_path, '{}.bin'.format(name)))

