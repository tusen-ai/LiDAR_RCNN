# LiDAR R-CNN: An Efficient and Universal 3D Object Detector

## Introduction

This is the official code of [LiDAR R-CNN: An Efficient and Universal 3D Object Detector](https://arxiv.org/abs/2103.15297). In this work, we present LiDAR R-CNN, a second stage detector that can generally improve any existing 3D detector. We find a common problem in Point-based RCNN, which is the learned features ignore the size of proposals, and propose several methods to remedy it. Evaluated on WOD benchmarks, our method significantly outperforms previous state-of-the-art.

中文介绍：https://zhuanlan.zhihu.com/p/359800738

## Requirements

 All the codes are tested in the following environment:

- Linux (tested on Ubuntu 16.04)
- Python 3.6+
- PyTorch 1.5 or higher (tested on PyTorch 1.5, 6, 7)
- CUDA 10.1

To install pybind11:

```shell
git clone git@github.com:pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake .. && make -j 
sudo make install
```

To install requirements:

```shell
pip install -r requirements.txt
apt-get install ninja-build libeigen3-dev
```

Install `LiDAR_RCNN` library:

```python
python setup.py develop --user
```

## Preparing Data

Please refer to [data processer](tools/data_processer/README.md) to generate the proposal data.

## Training

After preparing WOD data, we can train the vehicle only model in the paper, run this command:

```shell
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg config/lidar_rcnn.yaml --name lidar_rcnn
```

For 3 class in WOD:

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py --cfg config/lidar_rcnn_all_cls.yaml --name lidar_rcnn_all
```

The models and logs will  be saved to `work_dirs/outputs`. 

## Evaluation

To evaluate, run distributed testing with 4 gpus:

```sheel
python -m torch.distributed.launch --nproc_per_node=4 tools/test.py --cfg config/lidar_rcnn.yaml --checkpoint outputs/lidar_rcnn/checkpoint_lidar_rcnn_59.pth.tar
python tools/create_results.py --cfg config/lidar_rcnn.yaml
```

Note that, you should keep the `nGPUS`  in config equal to ` nproc_per_node` .This will generate a `val.bin` file in the `work_dir/results`. You can create submission to Waymo server using waymo-open-dataset code by following the instructions [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md).

## Results

Our model achieves the following performance on:

[Waymo Open Dataset Challenges (3D Detection)](https://waymo.com/open/challenges/2020/3d-detection/)

| Proposals from                                               | Class   | Channel | 3D AP L1 Vehicle | 3D AP L1 Pedestrian | 3D AP L1 Cyclist |
| ------------------------------------------------------------ | ------- | :-----: | :--------------: | :-----------------: | :--------------: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | Vehicle |   1x    |       75.6       |          -          |        -         |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | Vehicle |   2x    |       75.6       |          -          |        -         |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | 3 Class |   1x    |       73.4       |        70.7         |       67.4       |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | 3 Class |   2x    |       73.8       |        71.9         |       69.4       |

| Proposals from                                               | Class   | Channel | 3D AP L2 Vehicle | 3D AP L2 Pedestrian | 3D AP L2 Cyclist |
| ------------------------------------------------------------ | ------- | :-----: | :--------------: | :-----------------: | :--------------: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | Vehicle |   1x    |       66.8       |          -          |        -         |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | Vehicle |   2x    |       67.9       |          -          |        -         |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | 3 Class |   1x    |       64.8       |        62.4         |       64.8       |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) | 3 Class |   2x    |       65.1       |        63.5         |       66.8       |

## Citation

If you find our paper or repository useful, please consider citing

```tex
@article{li2021lidar,
  title={LiDAR R-CNN: An Efficient and Universal 3D Object Detector},
  author={Li, Zhichao and Wang, Feng and Wang, Naiyan},
  journal={CVPR},
  year={2021},
}
```

## Acknowledgement

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [LiDAR_SOT](https://github.com/TuSimple/LiDAR_SOT)
