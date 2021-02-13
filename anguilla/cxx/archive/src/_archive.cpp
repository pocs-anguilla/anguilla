// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// STL
#include <optional>

// Anguilla
#include <anguilla/archive/archive.hpp>
#include <anguilla/archive/individual.hpp>
#include <anguilla/archive/statistics.hpp>
namespace ag = anguilla;

typedef double f8;  // following Numba's convention

PYBIND11_MODULE(_archive, m) {
    m.doc() = "This module contains unbounded population archives.";

    py::class_<ag::upmo::Archive<f8>>(m, "UPMOArchive")
        .def(py::init<const ag::upmo::Parameters<f8> &, const std::optional<py::array_t<f8>> &>(),
             py::arg("parameters"),
             py::arg("reference") = std::nullopt)
        .def_property("size",
                      &ag::upmo::Archive<f8>::size,
                      nullptr)
        .def_property("empty",
                      &ag::upmo::Archive<f8>::empty,
                      nullptr)
        .def_property("nbytes",
                      &ag::upmo::Archive<f8>::nbytes,
                      nullptr)
        .def_property("reference",
                      &ag::upmo::Archive<f8>::reference,
                      nullptr,
                      py::return_value_policy::reference)
        .def_property("left_extreme",
                      &ag::upmo::Archive<f8>::leftExtremeIndividual,
                      nullptr,
                      py::return_value_policy::reference)
        .def_property("right_extreme",
                      &ag::upmo::Archive<f8>::rightExtremeIndividual,
                      nullptr,
                      py::return_value_policy::reference)
        .def("insert",
             &ag::upmo::Archive<f8>::insert,
             py::arg("point"),
             py::arg("fitness"),
             py::return_value_policy::reference)
        .def("sample_extreme",
             &ag::upmo::Archive<f8>::sampleExtreme,
             py::arg("p"),
             py::return_value_policy::reference)
        .def("sample_interior",
             &ag::upmo::Archive<f8>::sampleInterior,
             py::arg("p"),
             py::return_value_policy::reference)
        .def("nearest",
             &ag::upmo::Archive<f8>::nearest,
             py::arg("node"),
             py::return_value_policy::reference)
        .def(
            "__iter__", [](ag::upmo::Archive<f8> &archive) {
                // Based on: https://git.io/JtCsK
                return py::make_iterator(archive.begin(), archive.end());
            },
            py::keep_alive<0, 1>())
        .def("get_statistics", &ag::upmo::Archive<f8>::getStatistics)
        .def("merge", &ag::upmo::Archive<f8>::merge, py::arg("other"));

    py::class_<ag::upmo::Individual<f8>>(m, "UPMOIndividual")
        .def(py::init<const py::array_t<f8> &, const py::array_t<f8> &, f8, f8>(),
             py::arg("point"), py::arg("fitness"),
             py::arg("step_size") = 1.0,
             py::arg("p_succ") = 0.5)
        .def_readonly("point",
                      &ag::upmo::Individual<f8>::point,
                      py::return_value_policy::reference_internal)
        .def_readonly("fitness",
                      &ag::upmo::Individual<f8>::fitness,
                      py::return_value_policy::reference_internal)
        .def_readonly("contribution",
                      &ag::upmo::Individual<f8>::contribution)
        .def_readonly("acc_contribution",
                      &ag::upmo::Individual<f8>::accContribution)
        .def_readwrite("step_size",
                       &ag::upmo::Individual<f8>::step_size)
        .def_readwrite("p_succ",
                       &ag::upmo::Individual<f8>::p_succ)
        .def_property("cov",
                      &ag::upmo::Individual<f8>::getCov,
                      nullptr)
        .def("coord",
             &ag::upmo::Individual<f8>::coord,
             py::arg("index"));

    py::class_<ag::upmo::Statistics>(m, "UPMOStatistics")
        .def(py::init<>())
        .def_property("insert_success_ratio",
                      &ag::upmo::Statistics::getInsertSucessRatio,
                      nullptr);

    py::class_<ag::upmo::Parameters<f8>>(m, "UPMOParameters")
        .def(py::init<std::size_t, f8, std::optional<f8>, std::optional<f8>, std::optional<f8>, f8, std::optional<f8>, f8, f8, f8, std::optional<f8>>(),
             py::arg("n_dimensions"),
             py::arg("initial_step_size"),
             py::arg("d") = ag::upmo::DEF_D,
             py::arg("p_target_succ") = ag::upmo::DEF_P_TARGET_SUCC,
             py::arg("c_p") = ag::upmo::DEF_C_P,
             py::arg("p_threshold") = ag::upmo::DEF_P_THRESHOLD,
             py::arg("c_cov") = ag::upmo::DEF_C_COV,
             py::arg("p_extreme") = ag::upmo::DEF_P_EXTREME,
             py::arg("sigma_min") = ag::upmo::DEF_SIGMA_MIN,
             py::arg("alpha") = ag::upmo::DEF_ALPHA,
             py::arg("c_r") = ag::upmo::DEF_C_R,
             ag::upmo::PARAMETERS_DOCSTRING)
        .def_readonly("initial_step_size",
                      &ag::upmo::Parameters<f8>::initialStepSize)
        .def_readonly("d",
                      &ag::upmo::Parameters<f8>::d)
        .def_readonly("p_target_succ",
                      &ag::upmo::Parameters<f8>::pTargetSucc)
        .def_readonly("c_p",
                      &ag::upmo::Parameters<f8>::cP)
        .def_readonly("p_threshold",
                      &ag::upmo::Parameters<f8>::pThreshold)
        .def_readonly("c_cov",
                      &ag::upmo::Parameters<f8>::cCov)
        .def_readonly("p_extreme",
                      &ag::upmo::Parameters<f8>::pExtreme)
        .def_readonly("sigma_min",
                      &ag::upmo::Parameters<f8>::sigmaMin)
        .def_readonly("alpha",
                      &ag::upmo::Parameters<f8>::alpha)
        .def_readonly("c_r",
                      &ag::upmo::Parameters<f8>::cR);

#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
