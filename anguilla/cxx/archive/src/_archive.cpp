// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// STL
#include <optional>

// Anguilla
#include <anguilla/archive/archive.hpp>
#include <anguilla/archive/archive_kd.hpp>
#include <anguilla/archive/individual.hpp>

typedef double f8;  // following Numba's convention

PYBIND11_MODULE(_archive, m) {
    m.doc() = "This module contains unbounded population archives.";

    py::class_<archive::Archive<f8>>(m, "UPMOArchive")
        .def(py::init<const archive::Parameters<f8> &, const std::optional<py::array_t<f8>> &>(),
             py::arg("parameters"),
             py::arg("reference") = std::nullopt)
        .def_property("size", &archive::Archive<f8>::size, nullptr)
        .def_property("empty", &archive::Archive<f8>::empty, nullptr)
        .def_property("nbytes", &archive::Archive<f8>::nbytes, nullptr)
        .def_property("reference", &archive::Archive<f8>::reference, nullptr, py::return_value_policy::reference)
        .def_property("left_exterior", &archive::Archive<f8>::leftExterior, nullptr, py::return_value_policy::reference)
        .def_property("right_exterior", &archive::Archive<f8>::rightExterior, nullptr, py::return_value_policy::reference)
        .def("insert", &archive::Archive<f8>::insert, py::arg("point"), py::arg("fitness"), py::return_value_policy::reference)
        .def("sample_exterior", &archive::Archive<f8>::sampleExterior,
             py::arg("p"), py::return_value_policy::reference)
        .def("sample_interior", &archive::Archive<f8>::sampleInterior,
             py::arg("p"), py::return_value_policy::reference)
        .def("nearest", &archive::Archive<f8>::nearest,
             py::arg("node"), py::return_value_policy::reference)
        .def(
            "__iter__", [](archive::Archive<f8> &archive) {
                // Based on: https://git.io/JtCsK
                return pybind11::make_iterator(archive.begin(), archive.end());
            },
            pybind11::keep_alive<0, 1>())
        .def("prev", &archive::Archive<f8>::prev, py::arg("iterator"), pybind11::keep_alive<0, 1>())
        .def("next", &archive::Archive<f8>::next, py::arg("iterator"), pybind11::keep_alive<0, 1>())
        .def("merge", &archive::Archive<f8>::merge, py::arg("other"));

    py::class_<archive::Individual<f8>>(m, "UPMOIndividual")
        .def(py::init<const py::array_t<f8> &, const py::array_t<f8> &, f8, f8>(),
             py::arg("point"), py::arg("fitness"), py::arg("step_size") = 1.0, py::arg("p_succ") = 0.5)
        .def_readonly("point", &archive::Individual<f8>::point)
        .def_readonly("fitness", &archive::Individual<f8>::fitness)
        .def_readonly("contribution", &archive::Individual<f8>::contribution)
        .def_readonly("acc_contribution", &archive::Individual<f8>::accContribution)
        .def_readwrite("step_size", &archive::Individual<f8>::step_size)
        .def_readwrite("p_succ", &archive::Individual<f8>::p_succ)
        .def_property("cov", &archive::Individual<f8>::getCov, nullptr)
        .def("coord", &archive::Individual<f8>::coord, py::arg("index"));

    py::class_<archive::Parameters<f8>>(m, "UPMOParameters")
        .def(py::init<std::size_t, f8, std::optional<f8>, std::optional<f8>, std::optional<f8>, f8, std::optional<f8>, f8, f8, f8, std::optional<f8>>(),
             py::arg("n_dimensions"),
             py::arg("initial_step_size"),
             py::arg("d") = std::nullopt,
             py::arg("p_target_succ") = std::nullopt,
             py::arg("c_p") = std::nullopt,
             py::arg("p_threshold") = 0.44,
             py::arg("c_cov") = std::nullopt,
             py::arg("p_extreme") = 1.0 / 5.0,
             py::arg("sigma_min") = 1e-15,
             py::arg("alpha") = 3.0,
             py::arg("c_r") = std::nullopt, R"pbdoc(
                Parameters for UP-MO-CMA-ES.

                Parameters
                ----------
                n_dimensions
                    Dimensionality of the search space.
                initial_step_size
                    Initial step size.
                n_offspring: optional
                    Number of offspring per parent.
                d: optional
                    Step size damping parameter.
                p_target_succ: optional
                    Target success probability.
                c_p: optional
                    Success rate averaging parameter.
                p_threshold: optional
                    Smoothed success rate threshold.
                c_cov: optional
                    Covariance matrix learning rate.
                p_extreme: optional
                    Extreme point probability.
                sigma_min: optional
                    Selected point convergence threshold.
                alpha: optional
                    Interior point probability weight.
                c_r: optional
                    Covariance matrix recombination learning rate.

                Notes
                -----
                Implements the default values defined in Table 1, p. 3 \
                :cite:`2016:up-mo-cma-es`, Table 1, p. 5 :cite:`2007:mo-cma-es` \
                and p. 489 :cite:`2010:mo-cma-es`.)pbdoc")
        .def_readonly("initial_step_size", &archive::Parameters<f8>::initialStepSize)
        .def_readonly("d", &archive::Parameters<f8>::d)
        .def_readonly("p_target_succ", &archive::Parameters<f8>::pTargetSucc)
        .def_readonly("c_p", &archive::Parameters<f8>::cP)
        .def_readonly("p_threshold", &archive::Parameters<f8>::pThreshold)
        .def_readonly("c_cov", &archive::Parameters<f8>::cCov)
        .def_readonly("p_extreme", &archive::Parameters<f8>::pExtreme)
        .def_readonly("sigma_min", &archive::Parameters<f8>::sigmaMin)
        .def_readonly("alpha", &archive::Parameters<f8>::alpha)
        .def_readonly("c_r", &archive::Parameters<f8>::cR);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
