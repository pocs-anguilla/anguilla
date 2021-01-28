// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// STL
#include <optional>

// Anguilla
#include <anguilla/dominance/dominance.hpp>

typedef double f8;  // following Numba's convention

PYBIND11_MODULE(_dominance, m) {
    m.doc() = "This module contains operators related to Pareto dominance.";

    m.def("non_dominated_sort_f8", &dominance::nonDominatedSort<f8>,
          dominance::docstring,
          py::arg("points"),
          py::arg("max_rank") = std::nullopt);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
