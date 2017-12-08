#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <saxpy.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_saxpy, m) {

    m.def("znorm", &znorm);

}
