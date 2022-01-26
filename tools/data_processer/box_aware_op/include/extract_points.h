#ifndef EXTRACT_POINTS_H
#define EXTRACT_POINTS_H

#include <pybind11/eigen.h>
#include <math.h>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

MatrixXb extract_points(const py::EigenDRef<Eigen::MatrixXf> pc,
                        const py::EigenDRef<Eigen::VectorXf> bbox,
                        float expand_x=0.4f, float expand_y=0.4f, bool canonic=false);

#endif //EXTRACT_POINTS_H