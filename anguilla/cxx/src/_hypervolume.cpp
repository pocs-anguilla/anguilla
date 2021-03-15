// STL
#include <optional>

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Xtensor
#include <xtensor/xtensor.hpp>

// PyXtensor
#include <pyxtensor/pyxtensor.hpp>

// Anguilla
#include <anguilla/hypervolume/hv2d.hpp>
#include <anguilla/hypervolume/hv3d.hpp>
#include <anguilla/hypervolume/hvc2d.hpp>
#include <anguilla/hypervolume/hvc3d.hpp>
#include <anguilla/hypervolume/hvkd.hpp>

typedef double f8; // following Numba's convention

[[nodiscard]] auto
hv2d_f8(const xt::xtensor<f8, 2U>& points,
        const std::optional<xt::xtensor<f8, 1U>>& reference = std::nullopt,
        const bool ignoreDominated = false) {
    return hv2d::calculate<f8>(points, reference, ignoreDominated);
}

[[nodiscard]] auto
hvc2d_f8(const xt::xtensor<f8, 2U>& points,
         const std::optional<xt::xtensor<f8, 1U>>& reference = std::nullopt) {
    if (reference.has_value()) {
        return hvc2d::contributionsWithRef<f8>(points, *reference);
    }
    return hvc2d::contributions<f8>(points);
}

[[nodiscard]] auto
hv3d_f8(const xt::xtensor<f8, 2U>& points,
        const std::optional<xt::xtensor<f8, 1U>>& reference = std::nullopt,
        const bool ignoreDominated = false, const bool useBtree = true) {
    if (useBtree) {
        return hv3d::calculate<f8, hv3d::BTreeMap<f8>>(points, reference,
                                                       ignoreDominated);
    }
    return hv3d::calculate<f8, hv3d::RBTreeMap<f8>>(points, reference,
                                                    ignoreDominated);
}

[[nodiscard]] auto
hvc3d_f8(const xt::xtensor<f8, 2U>& points,
         const std::optional<xt::xtensor<f8, 1U>>& reference = std::nullopt,
         const bool useBtree = true, const bool preferExtrema = false) {
    if (useBtree) {
        return hvc3d::contributions<f8, hvc3d::BTreeMap<f8>>(points, reference,
                                                             preferExtrema);
    }
    return hvc3d::contributions<f8, hvc3d::RBTreeMap<f8>>(points, reference,
                                                          preferExtrema);
}

[[nodiscard]] auto
hvkd_f8(const xt::xtensor<f8, 2U>& points,
        const std::optional<xt::xtensor<f8, 1U>>& reference = std::nullopt,
        const bool ignoreDominated = false) {
    return hvkd::calculate<f8>(points, reference, ignoreDominated);
}

PYBIND11_MODULE(_hypervolume, m) {
    m.doc() = "Hypervolume algorithms implemented in C++.";

    m.def("hv2d_f8", &hv2d_f8, hv2d::docstring, py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("ignoreDominated") = false);

    m.def("hv3d_f8", &hv3d_f8, hv3d::docstring, py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("ignoreDominated") = false, py::arg("use_btree") = true);

    m.def("hvkd_f8", &hvkd_f8, hvkd::docstring, py::arg("points"),
          py::arg("reference") = std::nullopt,
          py::arg("ignoreDominated") = false);

    m.def("hvc2d_f8", &hvc2d_f8, hvc2d::docstring, py::arg("points"),
          py::arg("reference") = std::nullopt);

    m.def("hvc3d_f8", &hvc3d_f8, hvc3d::docstring, py::arg("points"),
          py::arg("reference") = std::nullopt, py::arg("use_btree") = true,
          py::arg("prefer_extrema") = false);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
