#include <pybind11/eigen.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "calc_conf.h"

Eigen::VectorXf get_pt_line_distance(const py::EigenDRef<Eigen::MatrixXf> pc, float a, float b, float c) {

  auto n = pc.block(0, 0, pc.rows(), 1).size();
  auto dis = (a * pc.block(0, 0, pc.rows(), 1) + b * pc.block(0, 1, pc.rows(), 1) + 
                      c * Eigen::VectorXf::Ones(n)).array().abs() / sqrt(a * a + b * b);
  return dis;
}

bool get_conf_single(const py::EigenDRef<Eigen::MatrixXf> pc,
              const py::EigenDRef<Eigen::MatrixXf> line_pt,
              const int STEP,
              const float MAX_DIS) {

  float x_len = fabs(line_pt(0, 0) - line_pt(1, 0));
  float y_len = fabs(line_pt(0, 1) - line_pt(1, 1));

  float a = line_pt(1, 1) - line_pt(0, 1);
  float b = line_pt(0, 0) - line_pt(1, 0);
  float c = line_pt(1, 0) * line_pt(0, 1) - line_pt(0, 0) * line_pt(1, 1);

  auto dis = get_pt_line_distance(pc, a, b, c);

  Eigen::MatrixXd::Index min_row, min_col;
  auto n = pc.block(0, 0, pc.rows(), 1).size();
  int cnt = 0;
  std::vector<bool> min_dis_flags(STEP);
  if(x_len > y_len) {

    float step = x_len / STEP;
    auto x_mat = line_pt.block(0, 0, 2, 1);
    float min_x = x_mat.minCoeff(&min_row, &min_col);    
    auto tmp_x = ((pc.block(0, 0, pc.rows(), 1) - min_x * Eigen::VectorXf::Ones(n)) / step).array().floor();
    for(int i = 0; i < n; ++i){
      if((dis(i) < MAX_DIS) && (tmp_x(i, 0) >= 0) && (tmp_x(i, 0) < STEP) && (min_dis_flags[int(tmp_x(i, 0))] == false)) {
        min_dis_flags[int(tmp_x(i, 0))] = true;
        cnt += 1;
      }
    }
  } else {

    float step = y_len / STEP;
    auto y_mat = line_pt.block(0, 1, 2, 1);
    float min_y = y_mat.minCoeff(&min_row, &min_col);
    auto tmp_y = ((pc.block(0, 1, pc.rows(), 1) - min_y * Eigen::VectorXf::Ones(n)) / step).array().floor();
    for(int i = 0; i < n; ++i){
      if((dis(i) < MAX_DIS) && (tmp_y(i, 0) >= 0) && (tmp_y(i, 0) < STEP) && (min_dis_flags[int(tmp_y(i, 0))] == false)) {
        min_dis_flags[int(tmp_y(i, 0))] = true;
        cnt += 1;
      }
    }
  }

  if(cnt > STEP / 1.5) return true;
  else return false;
}
