import yaml
import argparse
from easydict import EasyDict as edict
from LiDAR_RCNN.utils.eval_utils import merge_results, do_nms, create_bin

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                    help='experiment cfgure file name',
                    required=True,
                    type=str)
args = parser.parse_args()
cfg = edict(yaml.load(open(args.cfg, 'r')))

outputs_bboxes, outputs_socres = merge_results(cfg.TEST.TAT_PATH, cfg.nGPUS)
final_dets_dict = do_nms(outputs_bboxes, outputs_socres)
create_bin(final_dets_dict, cfg.TEST.TAT_PATH, cfg.TEST.FILE_NAME)