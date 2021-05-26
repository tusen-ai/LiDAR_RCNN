#include <pybind11/pybind11.h>
#include "calc_conf.h"
#include "overlap.h"
#include "polygon_overlap.h"
#include "extract_points.h"

namespace py = pybind11;

PYBIND11_MODULE(lidar_bbox_tools_c, m) {
  m.doc() = "Tusimple bbx conf calc";
  m.def("get_conf_single", &get_conf_single);
  m.def("overlap", &overlap);
  m.def("polygon_overlap", &polygon_overlap);
  m.def("extract_points", &extract_points);
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
