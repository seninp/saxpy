#include <pybind11/pybind11.h>

// std::vector<double> znorm(std::vector<double> ts, double threshold)
void init_znorm(py::module &);

// std::vector<double> subseries(std::vector<double> *ts, int start, int end)
void init_subseries(py::module &);

PYBIND11_MODULE(example, m) {
    init_znorm(m);
    init_subseries(m);
}
