import os
import re
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle as pkl
from collections import defaultdict
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from lidar_bbox_tools_c import extract_points, overlap, polygon_overlap


def get_proposal_dict(data, pc_path):
    outputs_dict = {}
    for o in data.objects:
        output = [
            o.object.box.center_x, o.object.box.center_y, o.object.box.center_z,
            o.object.box.length, o.object.box.width, o.object.box.height,
            o.object.box.heading, o.score, o.object.type
        ]
        key = "{}/{}".format(o.context_name, o.frame_timestamp_micros)
        if key not in outputs_dict:
            outputs_dict[key] = defaultdict(list)
        outputs_dict[key]['pred_lst'].append(output)
        outputs_dict[key]['pc_url'] = '{}/segment-{}_with_camera_labels/{}_1.npz'.format(
                pc_path, o.context_name, o.frame_timestamp_micros)
        outputs_dict[key]['pc_url_ri2'] = '{}/segment-{}_with_camera_labels/{}_2.npz'.format(
                pc_path, o.context_name, o.frame_timestamp_micros)
    return outputs_dict


def load_gt_npz(gt_path):
    gt_info = {}
    file_list = glob(os.path.join(gt_path, '*/*.npz'))
    for file_path in tqdm(file_list):
        data = np.load(file_path)
        context_pattern = re.compile(r'\d+_\d+_\d+_\d+_\d+')
        context = context_pattern.findall(file_path)[0]
        timestamp = os.path.basename(file_path).split('.')[0]
        key = "{}/{}".format(context, timestamp)
        if len(data['boxes']) != 0:
            boxes = data['boxes'][:, 0, :]
        else:
            boxes = data['boxes']
        gt_info[key] = {'boxes': boxes, 'ids': data['ids'], 'types': data['types'], 'pts_nums': data['pts_nums'], 'pose': data['pose'],'tss':data['tss']}
    return gt_info

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        result = pkl.load(f)
    return result

def read_bin(file_path):
    with open(file_path, 'rb') as f:
        objects = metrics_pb2.Objects()
        objects.ParseFromString(f.read())
    return objects

def get_matching_by_iou(pred_corners, gt_corners, valid_gt):
    iou_mat = np.zeros((pred_corners.shape[0], gt_corners.shape[0]), dtype=np.float32)
    polygon_overlap(pred_corners.reshape(-1, 8), gt_corners.reshape(-1, 8), iou_mat)
    matching_lst = np.argmax(iou_mat, axis=-1)
    cls_label = valid_gt[matching_lst, -1]
    return matching_lst, cls_label

def get_matching_gt(data, pred_list, min_points=5):
    if len(data['boxes']) == 0:
        return np.ones((len(pred_list), 9)), np.zeros(len(pred_list))
    gt_bbox = np.hstack([data['boxes'], np.array(data['types'])[:, None]])

    # only car, ped, cyc are inlcuded
    gt_class = np.array(data['types'])
    keep_1 = (gt_class == 1)
    keep_2 = (gt_class == 2)
    keep_4 = (gt_class == 4)

    keep_pts = (np.array(data['pts_nums']) > min_points)
    keep = (keep_1 | keep_2 | keep_4) & keep_pts
    valid_gt = gt_bbox[keep]
    if len(valid_gt) == 0:
        return np.ones((len(pred_list), 9)), np.zeros(len(pred_list))

    gt_corners = box_utils.get_upright_3d_box_corners(valid_gt[:, :-1]).numpy().astype(np.float32)
    pred_corners = box_utils.get_upright_3d_box_corners(pred_list[:, :7]).numpy().astype(np.float32)

    # only use BEV for simple
    gt_corners_2d = gt_corners[:, :4, :2]
    pred_corners_2d = pred_corners[:, :4, :2]

    matching_lst, cls_label = get_matching_by_iou(pred_corners_2d, gt_corners_2d, valid_gt)
    matching_gt_bbox = valid_gt[matching_lst, :].astype(np.float32)
    return matching_gt_bbox, cls_label

def get_objects_name(pc_url, idx):
    frame_name = pc_url.split('/')[-2].replace('segment-', '').replace('_with_camera_labels', '')
    ts = pc_url.split('/')[-1][:-6]
    name = frame_name + '/' + ts + '/' + str(idx)
    return name

def get_pc_w_trans(pc_url, ts, pose):
    root_path = "/".join(pc_url.split('/')[:-1])
    pc_url_ri1 = os.path.join(root_path, str(ts) + "_1.npz")
    pc_url_ri2 = os.path.join(root_path, str(ts) + "_2.npz")
    gt_url = os.path.join(root_path, str(ts) + ".npz").replace('pc', 'gt')
    gt_c = np.load(gt_url)
    pose_c = gt_c['pose']
    pcds_ri1 = np.load(pc_url_ri1)['pc']
    pcds_ri2 = np.load(pc_url_ri2)['pc']
    T = np.linalg.inv(pose) @ pose_c
    pcds_ri1_trans = (T[:3,:3] @ pcds_ri1.T).T + T[:3,3]
    pcds_ri2_trans = (T[:3,:3] @ pcds_ri2.T).T + T[:3,3]
    return pcds_ri1_trans.astype(np.float32), pcds_ri2_trans.astype(np.float32)

def add_frame_id(pcds, idx):
    pc_num = pcds.shape[0]
    pcds = np.hstack([pcds, np.full((pc_num, 1), idx)])
    return pcds

def process_single_frame(output_dict, score_thresh=0.0):
    # load data
    pred_lst = output_dict['pred_lst']
    expand_proposal_meter = output_dict['expand_proposal_meter']
    nframe = output_dict['nframe']
    tss = output_dict['tss']
    pose = output_dict['pose']

    if len(pred_lst) == 0:
        print('no pred here')
        return
    # filter proposals by score threshhold
    pred_lst = np.array(pred_lst, dtype=np.float32)
    valid_mask = pred_lst[:, 7] > score_thresh
    pred_lst = pred_lst[valid_mask]

    matching_gt_bbox, cls_label = get_matching_gt(output_dict, pred_lst)

    pcds_ri1 = []
    pcds_ri2 = []
    for frame_idx, ts in enumerate(tss[-1:-(nframe + 1):-1]):
        pcds_ri1_trans, pcds_ri2_trans = get_pc_w_trans(output_dict['pc_url'], ts, pose)
        pcds_ri1.append(pcds_ri1_trans)
        pcds_ri2.append(pcds_ri2_trans)

    # extract points
    for i in range(len(pred_lst)):
        # get centerx, centery, length, width, heading
        pred_bbox_bev = pred_lst[i, [0, 1, 3, 4, 6]].astype(np.float32)
        pcds_ri1_in_box_lst = []
        pcds_ri2_in_box_lst = []
        for frame_idx, ts in enumerate(tss[-1:-(nframe + 1):-1]):
            pcds_ri1_trans, pcds_ri2_trans = pcds_ri1[frame_idx], pcds_ri2[frame_idx]
            # enlarge the length and width by 3 meter
            valid_mask = extract_points(pcds_ri1_trans, pred_bbox_bev, expand_proposal_meter, expand_proposal_meter, False).reshape(-1)
            pcds_ri1_in_box = add_frame_id(pcds_ri1_trans[valid_mask], frame_idx).astype(np.float16)
            valid_mask_ri2 = extract_points(pcds_ri2_trans, pred_bbox_bev, expand_proposal_meter, expand_proposal_meter, False).reshape(-1)
            pcds_ri2_in_box = add_frame_id(pcds_ri2_trans[valid_mask_ri2], frame_idx).astype(np.float16)

            if (len(pcds_ri1_in_box) != 0 or len(pcds_ri2_in_box) != 0):
                pcds_ri1_in_box_lst.append(pcds_ri1_in_box)
                pcds_ri2_in_box_lst.append(pcds_ri2_in_box)
        if (len(pcds_ri1_in_box_lst) != 0 or len(pcds_ri2_in_box_lst) != 0):
            name = get_objects_name(output_dict['pc_url'], i)
            data = [pcds_ri1_in_box_lst, pcds_ri2_in_box_lst, pred_lst.astype(np.float16)[i], matching_gt_bbox[i].astype(np.float16), cls_label.astype(np.float16)[i]]
            data_byte = pkl.dumps(data)
            yield [name.encode('ascii'), data_byte]

