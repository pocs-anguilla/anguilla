// PyBind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Xtensor
#include <xtensor/xtensor.hpp>

// PyXtensor
#include <pyxtensor/pyxtensor.hpp>

typedef double f8;

// Anguilla
#include <anguilla/optimizers/mocma.hpp>
#include <anguilla/optimizers/selection.hpp>
namespace ag = anguilla;

PYBIND11_MODULE(_optimizers, m) {
    // xt::import_numpy(); // xtensor-python

    m.doc() = "This module contains implementations of algorithms for "
              "multi-objective optimization.";

    m.def("least_contributors", &ag::hvi::leastContributors<f8>,
          py::arg("points"), py::arg("k"));

    m.def("selection", &ag::hvi::selection<f8>, py::arg("points"),
          py::arg("ranks"), py::arg("targetSize"));

    m.def("cholesky_update", &ag::optimizers::choleskyUpdate<f8>, py::arg("L"),
          py::arg("alpha"), py::arg("beta"), py::arg("v"));

    py::enum_<ag::optimizers::SuccessNotion>(
        m, "SuccessNotion", ag::optimizers::MO_SUCCESS_NOTION_DOCSTRING)
        .value("PopulationBased",
               ag::optimizers::SuccessNotion::PopulationBased)
        .value("IndividualBased",
               ag::optimizers::SuccessNotion::IndividualBased)
        .export_values();

    py::class_<ag::optimizers::MOParameters<f8>>(
        m, "MOParameters", ag::optimizers::MO_PARAMETERS_DOCSTRING)
        .def(py::init<std::size_t, f8, std::size_t, std::optional<f8>,
                      std::optional<f8>, std::optional<f8>, f8,
                      std::optional<f8>, std::optional<f8>>(),
             py::arg("n_dimensions"), py::arg("initial_step_size") = 1.0,
             py::arg("n_offspring") = 1, py::arg("d") = std::nullopt,
             py::arg("p_target_succ") = std::nullopt,
             py::arg("c_p") = std::nullopt, py::arg("p_threshold") = 0.44,
             py::arg("c_c") = std::nullopt, py::arg("c_cov") = std::nullopt)
        .def_readonly("n_dimensions",
                      &ag::optimizers::MOParameters<f8>::nDimensions)
        .def_readonly("n_offspring",
                      &ag::optimizers::MOParameters<f8>::nOffspringPerParent)
        .def_readonly("initial_step_size",
                      &ag::optimizers::MOParameters<f8>::initialStepSize)
        .def_readonly("d", &ag::optimizers::MOParameters<f8>::d)
        .def_readonly("p_target_succ",
                      &ag::optimizers::MOParameters<f8>::pTargetSucc)
        .def_readonly("c_p", &ag::optimizers::MOParameters<f8>::cP)
        .def_readonly("p_threshold",
                      &ag::optimizers::MOParameters<f8>::pThreshold)
        .def_readonly("c_c", &ag::optimizers::MOParameters<f8>::cC)
        .def_readonly("c_cov", &ag::optimizers::MOParameters<f8>::cCov);

    py::class_<ag::optimizers::MOStoppingConditions<f8>>(
        m, "MOStoppingConditions",
        ag::optimizers::MO_STOPPING_CONDITIONS_DOCSTRING)
        .def(py::init<std::optional<std::size_t>, std::optional<std::size_t>,
                      std::optional<f8>, bool, bool>(),
             py::arg("max_generations") = std::nullopt,
             py::arg("max_evaluations") = std::nullopt,
             py::arg("target_indicator_value") = std::nullopt,
             py::arg("triggered") = false, py::arg("is_output") = false)
        .def_readonly("max_generations",
                      &ag::optimizers::MOStoppingConditions<f8>::maxGenerations)
        .def_readonly("max_evaluations",
                      &ag::optimizers::MOStoppingConditions<f8>::maxEvaluations)
        .def_readonly(
            "target_indicator_value",
            &ag::optimizers::MOStoppingConditions<f8>::targetIndicatorValue)
        .def_readonly("triggered",
                      &ag::optimizers::MOStoppingConditions<f8>::triggered)
        .def_readonly("is_output",
                      &ag::optimizers::MOStoppingConditions<f8>::isOutput);

    py::class_<ag::optimizers::MOSolution<f8>>(
        m, "MOSolution", ag::optimizers::MO_SOLUTION_DOCSTRING)
        .def(py::init<xt::xtensor<f8, 2U>, xt::xtensor<f8, 2U>,
                      xt::xtensor<f8, 2U>>(),
             py::arg("point"), py::arg("fitness"), py::arg("step_size"))
        .def_readonly("point", &ag::optimizers::MOSolution<f8>::point,
                      py::return_value_policy::reference_internal)
        .def_readonly("fitness", &ag::optimizers::MOSolution<f8>::fitness,
                      py::return_value_policy::reference_internal)
        .def_readonly("step_size", &ag::optimizers::MOSolution<f8>::stepSize,
                      py::return_value_policy::reference_internal);

    py::class_<ag::optimizers::MOCMA<f8>>(m, "MOCMA",
                                          ag::optimizers::MOCMA_DOCSTRING)
        .def(py::init<xt::xtensor<f8, 2U>, xt::xtensor<f8, 2U>,
                      std::optional<std::size_t>, f8, std::string,
                      std::optional<std::size_t>, std::optional<std::size_t>,
                      std::optional<f8>,
                      std::optional<ag::optimizers::MOParameters<f8>>,
                      std::optional<ag::optimizers::MOCMA<f8>::SeedType>>(),
             py::arg("parent_points"), py::arg("parent_fitness"),
             py::arg("n_offspring") = std::nullopt,
             py::arg("initial_step_size") = 1.0,
             py::arg("success_notion") = std::string("population"),
             py::arg("max_generations") = std::nullopt,
             py::arg("max_evaluations") = std::nullopt,
             py::arg("target_indicator_value") = std::nullopt,
             py::arg("parameters") = std::nullopt,
             py::arg("seed") = std::nullopt)
        .def("ask", &ag::optimizers::MOCMA<f8>::ask)
        .def("tell", &ag::optimizers::MOCMA<f8>::tell, py::arg("fitness"),
             py::arg("penalized_fitness") = std::nullopt,
             py::arg("evaluation_count") = std::nullopt)
        .def_property("population", &ag::optimizers::MOCMA<f8>::population,
                      nullptr)
        .def_property("success_notion",
                      &ag::optimizers::MOCMA<f8>::successNotion, nullptr)
        .def_property("name", &ag::optimizers::MOCMA<f8>::name, nullptr)
        .def_property("parameters", &ag::optimizers::MOCMA<f8>::parameters,
                      nullptr)
        .def_property("stop", &ag::optimizers::MOCMA<f8>::stop, nullptr)
        .def_property("best", &ag::optimizers::MOCMA<f8>::best, nullptr)
        .def_property("stopping_conditions",
                      &ag::optimizers::MOCMA<f8>::stoppingConditions, nullptr)
        .def_property("qualified_name",
                      &ag::optimizers::MOCMA<f8>::qualifiedName, nullptr)
        .def_property("generation_count",
                      &ag::optimizers::MOCMA<f8>::generationCount, nullptr)
        .def_property("evaluation_count",
                      &ag::optimizers::MOCMA<f8>::evaluationCount, nullptr);

    py::class_<ag::optimizers::MOPopulation<f8>>(m, "MOPopulation")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t, f8,
                      f8>(),
             py::arg("n_parents"), py::arg("n_offspring"),
             py::arg("n_dimensions"), py::arg("n_objectives"),
             py::arg("step_size"), py::arg("p_succ"))
        .def_readonly("point", &ag::optimizers::MOPopulation<f8>::point,
                      py::return_value_policy::reference_internal)
        .def_readonly("fitness", &ag::optimizers::MOPopulation<f8>::fitness,
                      py::return_value_policy::reference_internal)
        .def_readonly("penalized_fitness",
                      &ag::optimizers::MOPopulation<f8>::penalizedFitness,
                      py::return_value_policy::reference_internal)
        .def_readonly("step_size", &ag::optimizers::MOPopulation<f8>::stepSize,
                      py::return_value_policy::reference_internal)
        .def_readonly("p_succ", &ag::optimizers::MOPopulation<f8>::pSucc,
                      py::return_value_policy::reference_internal)
        .def_readonly("cov", &ag::optimizers::MOPopulation<f8>::cov,
                      py::return_value_policy::reference_internal)
        .def_readonly("last_z", &ag::optimizers::MOPopulation<f8>::lastZ,
                      py::return_value_policy::reference_internal)
        .def_readonly("last_step", &ag::optimizers::MOPopulation<f8>::lastStep,
                      py::return_value_policy::reference_internal)
        .def_readonly("parent_index",
                      &ag::optimizers::MOPopulation<f8>::parentIdx,
                      py::return_value_policy::reference_internal);
#ifdef VERSION_INFO
    m.attr("__version__") = Py_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
