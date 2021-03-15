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

#include <anguilla/common/pyshark.hpp>

static constexpr const char* _hvkd_docstring =
    "Calculate the hypervolume using Shark's implementation.";

static constexpr const char* _hvckd_docstring =
    "Calculate the hypervolume contributions using Shark's implementation.";

auto hvkd_f8(const PySharkVectorList<double>& ps,
             const PySharkVector<double>& ref_p,
             const bool use_approximation = false) {
    shark::HypervolumeCalculator hv;
    if (use_approximation) {
        hv.useApproximation(true);
    }

    return hv(ps, ref_p);
}

auto hvckd_f8(const PySharkVectorList<double>& ps,
              const PySharkVector<double>& ref_p,
              const bool use_approximation = false) {
    shark::HypervolumeContribution hvckd;
    if (use_approximation) {
        hvckd.useApproximation(true);
    }

    auto contributionPairs = hvckd.smallest(ps, ps.size(), ref_p);
    const auto n = contributionPairs.size();
    auto contribution = new double[n]();
    for (const auto [contrib, index] : contributionPairs) {
        contribution[index] = contrib;
    }

    py::capsule freeContributionsMemory(contribution, [](void* ptr) {
        std::unique_ptr<double[]>(static_cast<decltype(contribution)>(ptr));
    });

    const auto output =
        py::array_t<double>({n},              // shape
                            {sizeof(double)}, // stride
                            contribution, freeContributionsMemory);
    return output;
}

PYBIND11_MODULE(_shark_hypervolume, m) {
    m.doc() = "Bindings for Shark's HV implementations.";

    m.def("hvkd_f8", &hvkd_f8, _hvkd_docstring, py::arg("ps"), py::arg("ref_p"),
          py::arg("use_approximation") = false);

    m.def("hvckd_f8", &hvckd_f8, _hvckd_docstring, py::arg("ps"),
          py::arg("ref_p"), py::arg("use_approximation") = false);
}
