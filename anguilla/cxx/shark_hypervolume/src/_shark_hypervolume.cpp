#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeContribution.h>

#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "pyshark.hpp"

static constexpr const char *_hvkd_docstring =
    "Calculate the hypervolume using Shark's implementation.";

static constexpr const char *_hvckd_docstring =
    "Calculate the hypervolume contributions using Shark's implementation.";

double hvkd_f8(const PySharkVectorList<double> &ps,
               const PySharkVector<double> &ref_p,
               const bool use_approximation = false) {
    shark::HypervolumeCalculator hv;
    if (use_approximation) {
        hv.useApproximation(true);
    }

    return hv(ps, ref_p);
}

py::array_t<double> hvckd_f8(const PySharkVectorList<double> &ps,
                             const PySharkVector<double> &ref_p,
                             const bool use_approximation = false) {
    shark::HypervolumeContribution hvckd;
    if (use_approximation) {
        hvckd.useApproximation(true);
    }

    auto contributionPairs = hvckd.smallest(ps, ps.size(), ref_p);
    auto contributionsPtr = new std::vector<double>(contributionPairs.size());
    auto &contributions = *contributionsPtr;
    for (const auto [contribution, index] : contributionPairs) {
        contributions[index] = contribution;
    }

    py::capsule freeContributionsMemory(contributionsPtr, [](void *ptr) {
        auto concretePtr = static_cast<decltype(contributionsPtr)>(ptr);
        delete concretePtr;
    });

    const auto output = py::array_t<double>({contributionsPtr->size()},
                                            {sizeof(double)},
                                            contributionsPtr->data(),
                                            freeContributionsMemory);
    return output;
}

PYBIND11_MODULE(_shark_hypervolume, m) {
    m.doc() = "Bindings for Shark's HV implementations.";

    m.def("hvkd_f8", &hvkd_f8, _hvkd_docstring,
          py::arg("ps"), py::arg("ref_p"),
          py::arg("use_approximation") = false);

    m.def("hvckd_f8", &hvckd_f8, _hvckd_docstring,
          py::arg("ps"), py::arg("ref_p"),
          py::arg("use_approximation") = false);
}
