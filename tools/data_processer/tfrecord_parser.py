""" Extract the point cloud sequences from the tfrecords"""
import argparse
import math
import numpy as np
import json
import os
import sys
import multiprocessing
from google.protobuf.descriptor import FieldDescriptor as FD
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from collections import defaultdict
from tqdm import tqdm

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='../../../datasets/waymo/validation/',
    help='the location of tfrecords')
parser.add_argument('--output_folder', type=str, default='../../../datasets/waymo/sot/',
    help='the location of raw pcs')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)


def pb2dict(obj):
    """
    Takes a ProtoBuf Message obj and convertes it to a dict.
    """
    adict = {}
    # if not obj.IsInitialized():
    #     return None
    for field in obj.DESCRIPTOR.fields:
        if not getattr(obj, field.name):
            continue
        if not field.label == FD.LABEL_REPEATED:
            if not field.type == FD.TYPE_MESSAGE:
                adict[field.name] = getattr(obj, field.name)
            else:
                value = pb2dict(getattr(obj, field.name))
                if value:
                    adict[field.name] = value
        else:
            if field.type == FD.TYPE_MESSAGE:
                adict[field.name] = [pb2dict(v) for v in getattr(obj, field.name)]
            else:
                adict[field.name] = [v for v in getattr(obj, field.name)]
    return adict


def bbox_dict2array(box_dict, metadata_dict):
    """transform box dict in waymo_open_format to array
    Args:
        box_dict ([dict]): waymo_open_dataset formatted bbox
    """
    speed_key = ['speed_x', 'speed_y', 'accel_x', 'accel_y']
    for key in speed_key:
        if key not in metadata_dict:
            print(metadata_dict)
            metadata_dict[key] = 0

    result = np.array([
        box_dict['center_x'],
        box_dict['center_y'],
        box_dict['center_z'],
        box_dict['length'],
        box_dict['width'],
        box_dict['height'],
        box_dict['heading'],
        metadata_dict['speed_x'],
        metadata_dict['speed_y'],
        metadata_dict['accel_x'],
        metadata_dict['accel_y'],
    ])
    return result

def main(data_folder, output_folder, multi_process_token=(0, 1)):
    tf_records = os.listdir(data_folder)
    tf_records = [x for x in tf_records if 'tfrecord' in x]
    tf_records = sorted(tf_records) 
    pc_folder = os.path.join(output_folder, 'pc', 'raw_pc')
    for record_index, tf_record_name in enumerate(tf_records):
        if record_index % multi_process_token[1] != multi_process_token[0]:
            continue
        print('starting for raw pc', record_index + 1, ' / ', len(tf_records), ' ', tf_record_name)
        FILE_NAME = os.path.join(data_folder, tf_record_name)
        dataset = tf.data.TFRecordDataset(FILE_NAME, compression_type='')
        segment_name = tf_record_name.split('.')[0]

        frame_num = 0
        pcs = dict()
        gt_info = dict()
        tss = []
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            context = frame.context.name
            ts = frame.timestamp_micros
            tss.append(ts)
            pose = np.array(frame.pose.transform).reshape(4,4)
            # name = str(context) + '/' + str(ts)
     
            # extract the points
            (range_images, camera_projections, range_image_top_pose) = \
                frame_utils.parse_range_image_and_camera_projection(frame)
            # 3d points in vehicle frame. 
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
            points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
            points_all = np.concatenate(points, axis=0)
            points_all_ri2 = np.concatenate(points_ri2, axis=0)

            laser_labels = frame.laser_labels
            frame_ids = list()
            frame_boxes = list()
            frame_types = list()
            frame_nums = list()
            for laser_label in laser_labels:
                id = laser_label.id
                box = laser_label.box
                metadata = laser_label.metadata
                metadata_dict = pb2dict(metadata)
                box_dict = pb2dict(box)
                box_array = bbox_dict2array(box_dict, metadata_dict)
                frame_boxes.append(box_array[np.newaxis, :])
                frame_ids.append(id)
                frame_types.append(laser_label.type)
                frame_nums.append(laser_label.num_lidar_points_in_box)
            gt_info = {'boxes': frame_boxes, 'ids': frame_ids, 'types': frame_types, 'pts_nums': frame_nums, 'pose': pose,'tss':tss[-10:]}

            frame_num += 1
            if frame_num % 10 == 0:
                print('Record {} / {} FNumber {:}'.format(record_index + 1, len(tf_records), frame_num))

            pc_folder = os.path.join(output_folder, 'pc', segment_name)
            gt_folder = os.path.join(output_folder, 'gt', segment_name)
            if not os.path.exists(pc_folder):
                os.makedirs(pc_folder)
            if not os.path.exists(gt_folder):
                os.makedirs(gt_folder)
            pc_path1 = os.path.join(pc_folder, str(ts) + "_1")
            pc_path2 = os.path.join(pc_folder, str(ts) + "_2")
            gt_path = os.path.join(gt_folder, str(ts))
            np.savez_compressed(pc_path1, pc=points_all)
            np.savez_compressed(pc_path2, pc=points_all_ri2)
            np.savez_compressed(gt_path, **gt_info)
        print('{:} frames in total'.format(frame_num))


if __name__ == '__main__':
    # multiprocessing accelerate the speed
    pool = multiprocessing.Pool(args.process)
    for token in range(args.process):
        result = pool.apply_async(main, args=(args.data_folder, args.output_folder, (token, args.process)))
    pool.close()
    pool.join()
    # main(args.data_folder, args.output_folder)

