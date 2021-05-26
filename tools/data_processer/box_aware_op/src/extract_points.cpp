#include "extract_points.h"
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <queue>
#include <Eigen/SVD>

MatrixXb extract_points(const py::EigenDRef<Eigen::MatrixXf> pc,
                        const py::EigenDRef<Eigen::VectorXf> bbox,
                        float expand, bool canonic) {
  int pc_num = pc.rows();
  float yaw = bbox(4);
  float cos_yaw = std::cos(yaw);
  float sin_yaw = std::sin(yaw);

  MatrixXb valid_mask(pc_num, 1);
  for (int i=0; i<pc_num;i++){
    float r_x = (pc(i, 0) - bbox(0)) * cos_yaw + (pc(i, 1) - bbox(1)) * sin_yaw;
    float r_y = (pc(i, 0) - bbox(0)) * (-sin_yaw) + (pc(i, 1) - bbox(1)) * cos_yaw;
    //&& (pc(i, 2) < 2.0f)
    if ((std::abs(r_x) < bbox(2)/2 + expand/2) && (std::abs(r_y) < bbox(3)/2 + expand/2)) {
      valid_mask(i, 0) = true;
    }
    else{
      valid_mask(i, 0) = false;
    }
  }
  return valid_mask;
}