#ifndef ANGUILLA_HYPERVOLUME_HVC2D_HPP
#define ANGUILLA_HYPERVOLUME_HVC2D_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>
namespace py = pybind11;

#include "common.hpp"

namespace hvc2d {

static constexpr const char *docstring =
    "Calculate the exact hypervolume contributions for a set of 2-D points.";

template <typename T = double>
py::array_t<T> contributions(const Point2DList<T> &points, const Point2D<T> &refPoint) {
    const auto n = points.size();

    // Here we allocate memory for the contributions, initialized to zero,
    // plus an additional element for the sentinel nodes.
    auto contributionsPtr = new std::vector<T>(n + 1U);
    auto &contributions = *contributionsPtr;

    py::capsule freeContributionsMemory(contributionsPtr, [](void *ptr) {
        auto concretePtr = static_cast<decltype(contributionsPtr)>(ptr);
        delete concretePtr;
    });

    // We work on a vector but will return a Numpy array.
    // It uses the vector's memory, which will be freed once the array
    // goes out of scope (handled by the py::capsule).
    const auto retval = py::array_t<T>({n},
                                       {sizeof(T)},
                                       contributionsPtr->data(),
                                       freeContributionsMemory);
    if (n == 0U) {
        return retval;
    }

    std::vector<std::pair<Point2D<T>, std::size_t>> input;
    input.reserve(n + 2U);

    // Add predecesor sentinel point.
    constexpr auto lowest = std::numeric_limits<T>::lowest();

    // Copy point set.
    input.push_back(std::move(std::make_pair(Point2D<T>(lowest, refPoint.y), n + 1U)));
    {
        auto i = 0U;
        for (const auto &p : points) {
            input.emplace_back(p, i++);
        }
    }

    // Sort along the x-axis in ascending order.
    std::sort(input.begin() + 1U, input.end(),
              [](auto const &l, auto const &r) { return l.first.x < r.first.x; });

    // Add succesor sentinel point.
    input.push_back(std::move(std::make_pair(Point2D<T>(refPoint.x, lowest), n + 1U)));

    // A working buffer for tracking repeated point mappings.
    std::vector<std::pair<std::size_t, std::size_t>> equalMappings;

    // Process the points
    for (std::size_t i = 1U, m = n + 1U, rl = n + 2U; i < m; ++i) {
        const auto [p, index] = input[i];
        const auto &left = input[i - 1U].first;

        if (!(p.y < left.y)) {  // p is dominated by left, skip it
            continue;
        }

        auto s = i + 1U;
        auto dominated = i;  // will skip dominated points
        while (s < rl && !(p.y > input[s].first.y)) {
            dominated = s;
            s += 1U;
        }
        auto &right = input[s].first;
        contributions[index] = (right.x - p.x) * (left.y - p.y);
        i = dominated;
    }

    return retval;
}

}  // namespace hvc2d

#endif  // ANGUILLA_HYPERVOLUME_HVC2D_HPP
