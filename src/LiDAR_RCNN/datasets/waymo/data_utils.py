import numpy as np
import pickle as pkl
from LiDAR_RCNN.utils.bbox_utils import get_3d_box, box3d_iou


def jitter(bbox, thres):
    if np.random.rand() < thres:
        return bbox
    range_config = [[0.2, 0.1, np.pi / 12, 0.7], [0.3, 0.15, np.pi / 12, 0.6],
                    [0.5, 0.15, np.pi / 9, 0.5], [0.8, 0.15, np.pi / 6, 0.3],
                    [1.0, 0.15, np.pi / 3, 0.2]]
    idx = np.random.randint(low=0, high=len(range_config), size=(1, ))[0]

    pos_shift = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][0]
    hwl_scale = ((np.random.rand(3) - 0.5) / 0.5) * range_config[idx][1] + 1.0
    angle_rot = ((np.random.rand(1) - 0.5) / 0.5) * range_config[idx][2]

    aug_box3d = np.concatenate(
        [bbox[0:3] + pos_shift, bbox[3:6] * hwl_scale, bbox[6:7] + angle_rot])
    return aug_box3d


def process_pcd(pcd, proposal, keep_num):
    if pcd.shape[0] != 0:
        # move pcd to proposal's center
        pcd[:, 2] -= proposal[2]
        choice = np.random.choice(pcd.shape[0], keep_num, replace=True)
        point_set = pcd[choice, :]
        # transform the points to pred box coordinate
        norm_xy = point_set[:, :2] - np.array([proposal[0], proposal[1]
                                               ]).reshape(1, 2)
        point_set[:, :2] = np.dot(norm_xy, rotz(proposal[-1]))
        point_set = np.hstack([
            point_set, -point_set[:, :3] + proposal[[3, 4, 5]] / 2,
            proposal[[3, 4, 5]] / 2 + point_set[:, :3]
        ])
    else:
        point_set = np.full((keep_num, 12), 0)
    return point_set


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s], [s, c]])


def norm_angle(angle):
    if angle < -np.pi * 1.5:
        angle = angle + 2 * np.pi
    if angle > np.pi * 1.5:
        angle = angle - 2 * np.pi
    return angle


def load_data(it, nframe):
    pcd_ri1_lst, pcd_ri2_lst, proposal, gt_box, gt_cls = pkl.loads(it['data'])

    pcd_cur_ri1 = pcd_ri1_lst[0].astype(np.single)
    pcd_cur_ri2 = pcd_ri2_lst[0].astype(np.single)
    pcd_add_ones_ri1 = np.ones((pcd_cur_ri1.shape[0], 2))
    pcd_add_ones_ri2 = np.ones((pcd_cur_ri2.shape[0], 2))
    pcd_add_ones_ri1[:, 1] = 0
    pcd_add_ones_ri2[:, 0] = 0
    pcd_cur_ri1 = np.hstack([pcd_cur_ri1, pcd_add_ones_ri1])
    pcd_cur_ri2 = np.hstack([pcd_cur_ri2, pcd_add_ones_ri2])
    pcd_cur = np.vstack([pcd_cur_ri1, pcd_cur_ri2])
    if len(pcd_ri1_lst) == 1:
        pcd_pre = np.zeros((1, pcd_cur.shape[1]))
    else:
        pcd_pre_ri1 = np.vstack(pcd_ri1_lst[1:nframe])
        pcd_pre_ri2 = np.vstack(pcd_ri2_lst[1:nframe])
        pcd_add_ones_ri1 = np.ones((pcd_pre_ri1.shape[0], 2))
        pcd_add_ones_ri2 = np.ones((pcd_pre_ri2.shape[0], 2))
        pcd_add_ones_ri1[:, 1] = 0
        pcd_add_ones_ri2[:, 0] = 0
        pcd_pre_ri1 = np.hstack([pcd_pre_ri1, pcd_add_ones_ri1])
        pcd_pre_ri2 = np.hstack([pcd_pre_ri2, pcd_add_ones_ri2])
        pcd_pre = np.vstack([pcd_pre_ri1, pcd_pre_ri2])

    proposal = proposal.astype(np.single)[:7]
    gt_box = gt_box.astype(np.single)[:7]
    return pcd_cur, pcd_pre, proposal, gt_box, gt_cls


def relabel_by_iou(proposal, gt_box, gt_cls, thresholds):
    if gt_cls != 0:
        # 1 ve 2 ped 4 cyc
        proposal_bbox = get_3d_box(proposal[[3, 4, 5]], proposal[6],
                                   proposal[[0, 1, 2]])
        gt_bbox = get_3d_box(gt_box[[3, 4, 5]], gt_box[6], gt_box[[0, 1, 2]])
        iou_3d, _ = box3d_iou(proposal_bbox, gt_bbox)
        if iou_3d < thresholds[int(gt_cls)]:
            gt_cls = 0
    return gt_cls


def get_heading_residual(gt_heading, proposal_heading):
    proposal_heading = proposal_heading % (2 * np.pi)
    angle_flip = (proposal_heading + np.pi) % (2 * np.pi)
    gt_heading = gt_heading % (2 * np.pi)
    heading_residual = norm_angle(gt_heading - proposal_heading)
    heading_residual_flip = norm_angle(gt_heading - angle_flip)
    return heading_residual_flip if np.abs(heading_residual) > np.abs(
        heading_residual_flip) else heading_residual
