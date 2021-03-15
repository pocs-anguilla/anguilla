#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVKD_HPP_
#define ANGUILLA_HYPERVOLUME_HVKD_HPP_

// STL
#include <numeric>
#include <vector>

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

// Anguilla
#include <anguilla/dominance/dominance.hpp>
namespace ag = anguilla;

/* REFERENCES

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/JtaJK

[2012:hypervolume_wfg]
L. While, L. Bradstreet, & L. Barone (2012). A Fast Way of Calculating Exact
Hypervolumes. IEEE Transactions on Evolutionary Computation, 16(1), 86–95.
*/

namespace hvkd {
/* Public interface */
static constexpr const char* docstring = R"_(
    Calculate the exact hypervolume indicator for a set of 2-D points.

    Parameters
    ----------
    points
        The point set of mutually non-dominated points.
    reference: optional
        The reference point. Otherwise computed as the component-wise \
        maximum over all the points.

    Returns
    -------
    float
        The hypervolume indicator.

    Notes
    -----
    Implements the basic version of the WFG algorithm \
    presented by :cite:`2012:hypervolume_wfg`. Incorporates aspects from \
    the implementation by :cite:`2008:shark` (URL: https://git.io/JtaJK). \
    It assumes minimization.)_";

template <typename T>
[[nodiscard]] T
calculate(const xt::xtensor<T, 2U>& points,
          const std::optional<xt::xtensor<T, 1U>>& inputReference,
          const bool ignoreDominated = false);

/* Private interface */
namespace internal {
template <typename T>
[[nodiscard]] T wfg(const xt::xtensor<T, 2U>& points,
                    const xt::xtensor<T, 1U>& reference);
}

template <typename T>
[[nodiscard]] T
calculate(const xt::xtensor<T, 2U>& points,
          const std::optional<xt::xtensor<T, 1U>>& inputReference,
          const bool ignoreDominated) {

    const std::size_t n = points.shape(0U);
    if (n == 0U) {
        return 0.0;
    }

    // Compute reference point
    const std::size_t d = points.shape(1U);
    const bool refGiven = inputReference.has_value();
    constexpr auto lowest = std::numeric_limits<T>::lowest();
    xt::xtensor<T, 1U> reference =
        refGiven ? *inputReference
                 : xt::full_like(xt::xtensor<T, 1U>::from_shape({d}), lowest);

    std::vector<std::size_t> index;
    index.reserve(n);

    for (auto i = 0U; i < n; ++i) {
        if (refGiven && ignoreDominated) {
            if (xt::all(xt::less(xt::row(points, i), reference))) {
                index.push_back(i);
            }
        } else {
            index.push_back(i);
        }
        if (!refGiven) {
            reference = xt::maximum(xt::row(points, i), reference);
        }
    }

    // Sort the points by their last component.
    const auto sortedIdx = xt::argsort(xt::col(points, -1));
    const auto sortedPoints = xt::view(points, xt::keep(sortedIdx));
    return internal::wfg<T>(sortedPoints, reference);
}

/* Private */
namespace internal {

template <typename T>
auto constexpr limitSet(const xt::xtensor<T, 2U>& points,
                        const xt::xtensor<T, 1U>& point) {
    return xt::maximum(points, point);
}

template <typename T>
[[nodiscard]] T wfg(const xt::xtensor<T, 2U>& points,
                    const xt::xtensor<T, 1U>& reference) {

    const std::size_t n = points.shape(0U);
    T vol = 0.0;

    // This base case is from Shark:
    if (n == 1U) {
        vol = xt::prod(reference - xt::row(points, 0U))();
        return vol;
    }

    // The WFG recursive calls are here:
    for (auto i = 0U; i < n; ++i) { // excluhv' in the WFG paper
        const xt::xtensor<T, 1U> currentPoint = xt::row(points, i);
        const xt::xtensor<T, 2U> lset = limitSet<T>(
            xt::view(points, xt::range(i + 1U, n), xt::all()), currentPoint);
        [[maybe_unused]] const auto [ranks, maxRank] =
            ag::dominance::nonDominatedSort<T>(lset, std::make_optional<>(1U));
        const auto ndsetIdx =
            xt::squeeze(xt::from_indices(xt::argwhere(xt::equal(ranks, 1U))));
        const auto ndset = xt::view(lset, xt::keep(ndsetIdx));
        const T boxVol = xt::prod(reference - currentPoint)();
        const T wfgVol = wfg<T>(ndset, reference);
        vol += boxVol - wfgVol;
    }
    return vol;
}
} // namespace internal
} // namespace hvkd

#endif // ANGUILLA_HYPERVOLUME_HVKD_HPP_
