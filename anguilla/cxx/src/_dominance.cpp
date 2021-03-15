// STL
#include <optional>

// PyBind11
#include <pybind11/stl.h>
namespace py = pybind11;

// Xtensor
#include <xtensor/xtensor.hpp>

// PyXtensor
#include <pyxtensor/pyxtensor.hpp>

// Anguilla
#include <anguilla/dominance/dominance.hpp>
#include <anguilla/dominance/nondominated_set_2d.hpp>
#include <anguilla/optimizers/individual.hpp>
namespace ag = anguilla;

typedef double f8; // following Numba's convention

[[nodiscard]] auto
non_dominated_sort_f8(const xt::xtensor<f8, 2U>& points,
                      const std::optional<std::size_t>& maxRank) {
    return ag::dominance::nonDominatedSort<f8>(points, maxRank);
}

PYBIND11_MODULE(_dominance, m) {
    // xt::import_numpy(); // xtensor-python
    m.doc() = "This module contains operators related to Pareto dominance.";

    m.def("non_dominated_sort_f8", &non_dominated_sort_f8,
          ag::dominance::NON_DOMINATED_SORT_DOCTSTRING, py::arg("points"),
          py::arg("max_rank") = std::nullopt);

    py::class_<ag::dominance::NonDominatedSet2D<f8>>(m, "NonDominatedSet2D")
        .def(py::init<>())
        .def_property("size", &ag::dominance::NonDominatedSet2D<f8>::size,
                      nullptr)
        .def_property("empty", &ag::dominance::NonDominatedSet2D<f8>::empty,
                      nullptr)
        .def_property("upper_bound",
                      &ag::dominance::NonDominatedSet2D<f8>::upperBound,
                      nullptr)
        .def("merge", &ag::dominance::NonDominatedSet2D<f8>::merge,
             py::arg("other"))
        .def("insert", &ag::dominance::NonDominatedSet2D<f8>::insert,
             py::arg("points"));

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
