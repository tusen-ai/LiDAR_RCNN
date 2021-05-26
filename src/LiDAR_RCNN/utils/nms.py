import numpy as np
from lidar_bbox_tools_c import polygon_overlap


class BboxHash:
    """
    store the bbox in a hash dictionary, filter the non-overlap bboxes according to the hash index
    """
    def __init__(self, x_scale, y_scale):
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.bbox_dic = {}

    def create_dic(self, bboxes):
        for i in range(bboxes.shape[0]):
            indexes = self._get_index(bboxes[i])
            for index in indexes:
                if index not in self.bbox_dic:
                    self.bbox_dic[index] = {i}
                else:
                    self.bbox_dic[index].add(i)

    def _get_index(self, bbox):
        indexes = []
        bbox_4point = np.array(bbox).reshape(4, 2)
        bbox_2point = np.zeros(4, dtype=np.int16)
        bbox_2point[:2] = np.floor(np.min(bbox_4point, axis=0) /
                                   self.x_scale).astype(np.int16)
        bbox_2point[2:4] = np.ceil(np.max(bbox_4point, axis=0) /
                                   self.y_scale).astype(np.int16)
        for i in range(bbox_2point[0], bbox_2point[2]):
            for j in range(bbox_2point[1], bbox_2point[3]):
                indexes.append(i * 100 + j)
        return indexes

    def get_filter_result(self, bbox):
        result = set()
        indexes = self._get_index(bbox)
        for index in indexes:
            if index in self.bbox_dic:
                result |= (self.bbox_dic[index])
        return result

    def clear_dic(self):
        self.bbox_dic = {}


def bbox_4point_overlaps(anchors, gt_boxes):
    """
    calculate the overlap of polygon
    :param anchors: 4point list, [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
    :param gt_boxes: 4point, [[x1, y1, x2, y2, x3, y3, x4, y4], ...]
    :return:
    """
    overlaps_ratios = np.zeros((anchors.shape[0], gt_boxes.shape[0]),
                               dtype=np.float32)
    polygon_overlap(anchors.astype(np.float32), gt_boxes.astype(np.float32),
                    overlaps_ratios)
    return overlaps_ratios


def wnms_wrapper(thresh_lo, thresh_hi, yaw_thre=0.3):
    def _nms(dets):
        return py_4pts_nms_hash_with_angle(dets, thresh_lo, thresh_hi,
                                           yaw_thre)

    return _nms


def py_4pts_nms_hash_with_angle(dets, thresh_lo, thresh_hi, yaw_thre=0.3):
    """
    voting boxes with confidence > thresh_hi
    keep boxes overlap <= thresh_lo
    rule out overlap > thresh_hi
    :param dets: [[x1, y1, x2, y2, x3, y3, x4,y4 score]]
    :param thresh_lo: retain overlap <= thresh_lo
    :param thresh_hi: vote overlap > thresh_hi
    :return: indexes to keep
    """
    dets_hash = BboxHash(100, 100)
    dets_hash.create_dic(dets[:, :8])
    scores = dets[:, -1]
    yaw = dets[:, 8]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        filter_indexes = list(dets_hash.get_filter_result(dets[i][:8]))
        in_mask = np.isin(order, filter_indexes)
        filter_order = order[in_mask]
        overlaps = bbox_4point_overlaps(dets[filter_order, :8],
                                        dets[i][None, :8])
        inds = np.where(overlaps > thresh_lo)[0]
        inds_keep = np.where(overlaps > thresh_hi)[0]
        if len(inds_keep) == 0:
            break
        order_keep = filter_order[inds_keep]
        # calculate the car face
        if order_keep.shape[0] <= 2:
            score_index = np.argmax(scores[order_keep])
            median = yaw[order_keep][score_index]
        elif order_keep.shape[0] % 2 == 0:
            tmp_yaw = yaw[order_keep].copy()
            tmp_yaw = np.append(tmp_yaw, yaw[order_keep[0]])
            median = np.median(tmp_yaw)
        else:
            median = np.median(yaw[order_keep])
        yaw_keep = np.where(
            abs(yaw[order_keep] - median) % (2 * np.pi) < yaw_thre)[0]
        order_keep = order_keep[yaw_keep]
        tmp = np.sum(scores[order_keep])
        bbox_avg = np.sum(scores[order_keep, None] * dets[order_keep, 0:11],
                          axis=0) / tmp
        keep.append(np.hstack((bbox_avg, [scores[i]])))
        order_delete = filter_order[inds]
        in_mask = np.isin(order, order_delete, invert=True)
        order = order[in_mask]
    dets_hash.clear_dic()
    return np.array(keep)
