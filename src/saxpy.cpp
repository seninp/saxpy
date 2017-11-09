#include <pybind11/pybind11.h>

// std::vector<double> znorm(std::vector<double> ts, double threshold)
void znorm(py::module &);

// std::vector<double> subseries(std::vector<double> *ts, int start, int end)
void subseries(py::module &);

PYBIND11_MODULE(example, m) {
    znorm(m);
    subseries(m);
}
