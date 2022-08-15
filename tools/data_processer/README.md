# Data Pipeline for LiDAR-RCNN in WOD

To decode points clouds and ground truth from Waymo Open Dataset.

```bash
cd tools/data_processer/
CUDA_VISIBLE_DEVICES='' python3 tfrecord_parser.py --data_folder /your/path/to/tfrecord --output_folder /your/path/to/save/processed_data --process 20
# CUDA_VISIBLE_DEVICES='' to disable GPU.
```

Then we get

```bash
ROOT_DIRECTORY
├── gt: 
      groud truth info.
├── pc:
      raw point cloud data.
```

After that, you can follow the [Tutorial](https://github.com/open-mmlab/mmdetection3d/blob/v0.13.0/docs/tutorials/waymo.md) in [mmdet3D(v0.13.0)](https://github.com/open-mmlab/mmdetection3d/tree/v0.13.0) to get proposals for LiDAR R-CNN.  Briefly, we can get proposals in **validation set** by:

```
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.py checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car-9fa20624.pth 8 --out results/waymo-pp-car/results_eval.pkl \
    --format-only --options 'pklfile_prefix=results/waymo-pp-car/kitti_results' \
    'submission_prefix=results/waymo-pp-car/kitti_results'
```

Regarding **training set**, we provide some tricky solutions. Maybe there are more reasonable way.

Firstly, we modify the codes in  [mmdet3D(v0.13.0)](https://github.com/open-mmlab/mmdetection3d/tree/v0.13.0)

```python
./configs/_base_/datasets/waymoD5-3d-car.py
Line 135 - ann_file=data_root + 'waymo_infos_val.pkl',
Line 135 + ann_file=data_root + 'waymo_infos_train.pkl',

./mmdet3d/core/evaluation/waymo_utils/prediction_kitti_to_waymo.py
Line 182 - filename = f'{self.prefix}{file_idx:03d}{frame_num:03d}'
Line 182 + filename = f'{file_idx}{frame_num:03d}'

./mmdet3d/datasets/waymo_dataset.py
Line 191 - waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
Line 192 - prefix = '1'
Line 191 + waymo_tfrecords_dir = osp.join(waymo_root, 'training')
Line 192 + prefix = '0'
```
and run
```
./tools/dist_test.sh configs/pointpillars/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car.py checkpoints/hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-car-9fa20624.pth 8 --out results/waymo-pp-car/results_eval.pkl \
    --format-only --options 'pklfile_prefix=results/waymo-pp-car/kitti_results_train' \
    'submission_prefix=results/waymo-pp-car/kitti_results_train'
```
Subsequently, set your data path in config and run ```data_processer.py ```

```yaml
num_process: 10
pc_path: /your/path/to/raw point cloud
data_path: /your/mmdet3d/waymo-car/kitti_results_train.bin
gt_path: /your/path/to/groud truth info.
target_path: datasets/mmdet3d_pp_5_frame
mode: train
expand_proposal_meter: 3
nframe: 5
```

```bash
CUDA_VISIBLE_DEVICES='' python3 data_processer.py config/mmdet3d_pp_train.yaml
CUDA_VISIBLE_DEVICES='' python3 data_processer.py config/mmdet3d_pp_val.yaml
```

After the data conversion, the folder structure should be organized as below.

```yaml
target_path
├── train.rec
├── train.idx
├── val.rec
├── val.idx
```

