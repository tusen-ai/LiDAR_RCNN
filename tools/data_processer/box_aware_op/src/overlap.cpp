#include "overlap.h"
#include <time.h>

void overlap(const py::EigenDRef<Eigen::MatrixXf> boxes1,
         const py::EigenDRef<Eigen::MatrixXf> boxes2,
         py::EigenDRef<Eigen::MatrixXf> iou_matrix) {
    OverlapChecker checker;
    checker.overlap(boxes1, boxes2, iou_matrix);
}