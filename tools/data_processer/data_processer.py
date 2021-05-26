import os
import sys
import yaml
import tfrecord
import data_utils
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multi_processer import MultiProcesser
from easydict import EasyDict as edict

fname_yaml = sys.argv[1]
with open(fname_yaml, 'r') as f:
    cfg = edict(yaml.load(f))

outputs = data_utils.read_bin(cfg.data_path)
print('proposal length:', len(outputs.objects))
outputs_dict = data_utils.get_proposal_dict(outputs, cfg.pc_path)
print('loading gt')
gt_dict = data_utils.load_gt_npz(cfg.gt_path)

data_list = []
for key in tqdm(outputs_dict):
    outputs_dict[key].update(gt_dict[key])
    outputs_dict[key]['expand_proposal_meter'] = cfg.expand_proposal_meter
    data_list.append(outputs_dict[key])

# It's essential for tfrecord.
np.random.shuffle(data_list)
print('The number of frames for training: ', len(data_list))

os.system('mkdir -p {}'.format(cfg.target_path))
target_file = os.path.join(cfg.target_path, cfg.mode)
record = tfrecord.TFRecordWriter('{}.rec'.format(target_file))
processer = MultiProcesser(data_list,
                           data_utils.process_single_frame,
                           num_workers=cfg.num_process)
for i, data in enumerate(processer.run()):
    name_byte, data_byte = data
    record.write({
        "name": (name_byte, "byte"),
        "data": (data_byte, "byte"),
    })
record.close()

os.system('python3 -m tfrecord.tools.tfrecord2idx {0} {1}'.format('{}.rec'.format(target_file), '{}.idx'.format(target_file)))

# single for debug
# record = tfrecord.TFRecordWriter('{}.rec'.format(target_file))
# for data in tqdm(data_list):
#     for name_byte, data_byte in data_utils.process_single_frame(data):
#         record.write({
#             "name": (name_byte, "byte"),
#             "data": (data_byte, "byte"),
#         })
# record.close()