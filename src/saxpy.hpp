#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <vector>

std::vector<double> znorm(std::vector<double> ts, double threshold);
