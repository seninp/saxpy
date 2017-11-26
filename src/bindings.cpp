#include "saxpy.hpp"

PYBIND11_MODULE(saxpy, m) {

    m.def("znorm", &znorm);

}
