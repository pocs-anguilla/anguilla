#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HV2D_HPP
#define ANGUILLA_HYPERVOLUME_HV2D_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL
#include <algorithm>
#include <vector>

// Anguilla
#include <anguilla/common/common.hpp>

/* REFERENCES

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm0V
*/

/* Public interface */
namespace hv2d {
static constexpr const char *docstring = R"_(
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
[[nodiscard]] auto calculate(const py::array_t<T> &, const std::optional<py::array_t<T>> &);
}  // namespace hv2d

/* Internal interface */
namespace __hv2d {
template <typename T>
[[nodiscard]] auto calculate(std::vector<Point2D<T>> &, const T, const T);
}  // namespace __hv2d

/* Public implementation */
namespace hv2d {
template <typename T>
auto calculate(const py::array_t<T> &_points, const std::optional<py::array_t<T>> &_reference) {
    static_assert(std::is_floating_point<T>::value,
                  "HV2D is not meant to be instantiated with non floating point type.");

    auto pointsR = _points.template unchecked<2>();
    assert(pointsR.shape(0) >= 0);
    const auto n = static_cast<std::size_t>(pointsR.shape(0));
    if (n == 0U) {
        return 0.0;
    }

    constexpr auto lowest = std::numeric_limits<T>::lowest();
    T refX = lowest;
    T refY = lowest;
    const bool refGiven = _reference.has_value();

    if (refGiven) {
        auto referenceR = _reference->template unchecked<1>();
        refX = referenceR(0);
        refY = referenceR(1);
    }

    std::vector<Point2D<T>> points;
    points.reserve(n);
    for (auto i = 0U; i < n; ++i) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        points.emplace_back(pX, pY);
        if (!refGiven) {
            refX = std::max(refX, pX);
            refY = std::max(refY, pY);
        }
    }

    return __hv2d::calculate<T>(points, refX, refY);
}
}  // namespace hv2d

/* Internal implementation */
namespace __hv2d {
template <typename T>
auto calculate(std::vector<Point2D<T>> &points, const T refX, const T refY) {
    const auto n = points.size();
    // Sort the point list along the x-axis in ascending order.
    std::sort(points.begin(), points.end(),
              [](auto const &l, auto const &r) { return l.x < r.x; });

    // Perform the integration
    T volume = 0.0;
    T lastY = refY;
    for (std::size_t i = 0U; i < n; ++i) {
        const auto &p = points[i];
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
}  // namespace __hv2d

#endif  // ANGUILLA_HYPERVOLUME_HV2D_HPP
