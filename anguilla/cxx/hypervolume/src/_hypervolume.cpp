#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hv2d.hpp"
#include "hv3d.hpp"
#include "hvc2d.hpp"
#include "hvc3d.hpp"

PYBIND11_MODULE(_hypervolume, m) {
    m.doc() = "Hypervolume algorithms implemented in C++.";

    m.def("hv2d_f8", &hv2d::calculate<>, hv2d::docstring,
          py::arg("ps"), py::arg("ref_p"));

    m.def("hv3d_btree_f8", &hv3d::calculate<>, hv3d::docstring,
          py::arg("ps"), py::arg("ref_p"));

    m.def("hv3d_rbtree_f8", &hv3d::calculate<double, hv3d::RBTreeMap<double>>,
          hv3d::docstring, py::arg("ps"), py::arg("ref_p"));

    m.def("hvc2d_f8", &hvc2d::contributions<>, hvc2d::docstring,
          py::arg("ps"), py::arg("ref_p"));

    m.def("hvc3d_btree_f8", &hvc3d::contributions<>, hvc3d::docstring,
          py::arg("ps"), py::arg("ref_p"));

    m.def("hvc3d_rbtree_f8", &hvc3d::contributions<double, hvc3d::RBTreeMap<double>>,
          hvc3d::docstring, py::arg("ps"), py::arg("ref_p"));

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
