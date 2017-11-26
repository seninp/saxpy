#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

std::vector<double> znorm(std::vector<double> ts, double threshold);
