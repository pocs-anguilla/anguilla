#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HV2D_HPP
#define ANGUILLA_HYPERVOLUME_HV2D_HPP

// STL
#include <algorithm>
#include <numeric>
#include <vector>

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xtensor.hpp>

// Anguilla
#include <anguilla/common/common.hpp>

/* REFERENCES

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm0V
*/

namespace hv2d {
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
    Ported from :cite:`2008:shark`.)_";

template <typename T>
[[nodiscard]] T
calculate(const xt::xtensor<T, 2U>& inputPoints,
          const std::optional<xt::xtensor<T, 1U>>& inputReference,
          const bool ignoreDominated = false);

/* Internal interface */
namespace internal {
template <typename T>
[[nodiscard]] T calculate(std::vector<Point2D<T>>& points, const T refX,
                          const T refY);
} // namespace internal

/* Public implementation */
template <typename T>
T calculate(const xt::xtensor<T, 2U>& inputPoints,
            const std::optional<xt::xtensor<T, 1U>>& inputReference,
            const bool ignoreDominated) {
    static_assert(
        std::is_floating_point<T>::value,
        "HV2D is not meant to be instantiated with non floating point type.");

    const std::size_t n = inputPoints.shape(0U);
    if (n == 0U) {
        return 0.0;
    }

    constexpr auto lowest = std::numeric_limits<T>::lowest();
    T refX = lowest;
    T refY = lowest;
    const bool refGiven = inputReference.has_value();

    if (refGiven) {
        refX = (*inputReference)[0U];
        refY = (*inputReference)[1U];
    }

    std::vector<Point2D<T>> points;
    points.reserve(n);
    for (auto i = 0U; i < n; ++i) {
        const auto pX = inputPoints(i, 0U);
        const auto pY = inputPoints(i, 1U);

        if (refGiven && ignoreDominated) {
            if (pX < refX && pY < refY) {
                points.emplace_back(pX, pY);
            }
        } else {
            points.emplace_back(pX, pY);
        }

        if (!refGiven) {
            refX = std::max(refX, pX);
            refY = std::max(refY, pY);
        }
    }

    return internal::calculate<T>(points, refX, refY);
}

/* Internal implementation */
namespace internal {
template <typename T>
T calculate(std::vector<Point2D<T>>& points, const T refX, const T refY) {
    const auto n = points.size();
    // Sort the point list along the x-axis in ascending order.
    std::sort(points.begin(), points.end(),
              [](auto const& l, auto const& r) { return l.x < r.x; });

    // Perform the integration
    T volume = 0.0;
    T lastY = refY;
    for (std::size_t i = 0U; i < n; ++i) {
        const auto& p = points[i];
        auto yDiff = lastY - p.y;
        // skip dominated points
        // point is dominated <=> y_diff <= 0
        if (yDiff > 0.0) {
            volume += (refX - p.x) * yDiff;
            lastY = p.y;
        }
    }
    return volume;
}
} // namespace internal
} // namespace hv2d

#endif // ANGUILLA_HYPERVOLUME_HV2D_HPP
