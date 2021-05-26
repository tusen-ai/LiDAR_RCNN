#ifndef CALC_CONF_H
#define CALC_CONF_H

#include <pybind11/eigen.h>
#include <cmath>

namespace py = pybind11;

bool get_conf_single(const py::EigenDRef<Eigen::MatrixXf> pc,
                     const py::EigenDRef<Eigen::MatrixXf> line_pt,
                     const int STEP,
                     const float MAX_DIS);

#endif // CALC_CONF_H