#ifndef ANGUILLA_HYPERVOLUME_HV2D_HPP
#define ANGUILLA_HYPERVOLUME_HV2D_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>
namespace py = pybind11;

#include "common.hpp"

namespace hv2d {

static constexpr const char *docstring =
    "Calculate the exact hypervolume indicator for a set of 2-D points.";

template <typename T = double>
T calculate(Point2DList<T> &points, const Point2D<T> &refPoint) {
    const auto n = points.size();

    if (n == 0U) {
        return 0.0;
    }

    // Copy point set and sort along the x-axis in ascending order.
    std::sort(points.begin(), points.end(),
              [](auto const &l, auto const &r) { return l.x < r.x; });

    // Perform the integration
    T volume = 0.0;
    T lastY = 0.0;
    {
        const auto &p = points[0U];
        lastY = p.y;
        volume += (refPoint.x - p.x) * (refPoint.y - lastY);
    }
    for (std::size_t i = 1U; i < n; ++i) {
        const auto &p = points[i];
        auto yDiff = lastY - p.y;
        // skip dominated points
        // point is dominated <=> y_diff <= 0
        if (yDiff > 0.0) {
            volume += (refPoint.x - p.x) * yDiff;
            lastY = p.y;
        }
    }

    return volume;
}

}  // namespace hv2d

#endif  // ANGUILLA_HYPERVOLUME_HV2D_HPP
