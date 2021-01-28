#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <anguilla/hypervolume/hv2d.hpp>
#include <anguilla/hypervolume/hv3d.hpp>
#include <anguilla/hypervolume/hvc2d.hpp>
#include <anguilla/hypervolume/hvc3d.hpp>

typedef double f8;  // following Numba's convention

auto hvc2d_f8(const py::array_t<f8>& points,
              const std::optional<py::array_t<f8>>& reference = std::nullopt,
              const bool nonDominated = true) {
    if (reference.has_value()) {
        return hvc2d::contributionsWithRef<f8>(points, reference.value(), nonDominated);
    }
    return hvc2d::contributions<f8>(points, nonDominated);
}

auto hv3d_f8(const py::array_t<f8>& points,
             const std::optional<py::array_t<f8>>& reference = std::nullopt,
             const bool useBtree = true) {
    if (useBtree) {
        return hv3d::calculate<f8, hv3d::BTreeMap<f8>>(points, reference);
    }
    return hv3d::calculate<f8, hv3d::RBTreeMap<f8>>(points, reference);
}

auto hvc3d_f8(const py::array_t<f8>& points,
              const std::optional<py::array_t<f8>>& reference = std::nullopt,
              const bool useBtree = true) {
    if (useBtree) {
        return hvc3d::contributions<f8, hvc3d::BTreeMap<f8>>(points, reference);
    }
    return hvc3d::contributions<f8, hvc3d::RBTreeMap<f8>>(points, reference);
}

PYBIND11_MODULE(_hypervolume, m) {
    m.doc() = "Hypervolume algorithms implemented in C++.";

    m.def("hv2d_f8", &hv2d::calculate<f8>,
          hv2d::docstring,
          py::arg("points"),
          py::arg("reference") = std::nullopt);

    m.def("hv3d_f8", &hv3d_f8,
          hv3d::docstring,
          py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("use_btree") = true);

    m.def("hvc2d_f8", &hvc2d_f8,
          hvc2d::docstring,
          py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("non_dominated") = true);

    m.def("hvc3d_f8", &hvc3d_f8,
          hvc3d::docstring,
          py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("use_btree") = true);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
