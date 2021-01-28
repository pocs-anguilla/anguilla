#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVC2D_NG_HPP
#define ANGUILLA_HYPERVOLUME_HVC2D_NG_HPP

// Pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Boost Intrusive
#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>

// STL
#include <algorithm>
#include <numeric>
#include <vector>

template <typename T>
struct IndexedPoint2D : public boost::intrusive::avl_set_base_hook<boost::intrusive::optimize_size<false>> {
    IndexedPoint2D(T pX, T pY, std::size_t index) : pX(pX), pY(pY), index(index) {}

    friend bool operator<(const IndexedPoint2D &l, IndexedPoint2D &r) { return l.pX < r.pX; }
    friend bool operator>(const IndexedPoint2D &l, const IndexedPoint2D &r) { return l.pX > r.pX; }
    friend bool operator==(const IndexedPoint2D &l, const IndexedPoint2D &r) { return l.pX == r.pX; }

    T pX;
    T pY;
    std::size_t index;
    boost::intrusive::avl_set_member_hook<> member_hook_;
};

template <typename T>
using Point2DIterator = std::vector<IndexedPoint2D<T>>::iterator;

template <typename T>
void contributions(const py::array_t<T> &points_, const std::optional<py::array_t<T>> &reference = std::nullopt, bool nonDominated = true) {
    static_assert(std::is_floating_point<T>::value,
                  "HVC2D is not meant to be instantiated with a non floating point type.");
    const auto pointsR = points_.template unchecked<2>();
    assert(pointsR.shape(0) >= 0);
    const auto n = static_cast<std::size_t>(pointsR.shape(0));
    if (n == 0U) {
        return py::array_t<T>(0U);
    }

    std::vector<IndexedPoint2D<T>> points;
    points.reserve(n + 2U);

    constexpr auto max = std::numeric_limits<T>::max();
    constexpr auto lowest = std::numeric_limits<T>::lowest();

    T refX = max;
    T refY = max;
    if (reference.has_value()) {
        auto referenceR = reference.value().template unchecked<1>();
        refX = referenceR(0);
        refY = referenceR(1);
    }
    points.emplace_back(lowest, refY, n);  // predecesor sentinel
    for (auto i = 0U; i < n; i++) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        points.emplace_back(pX, pY, i);
    }
    if (nonDominated) {
        std::sort(std::next(points.begin()), points.end());
        points.emplace_back(refX, lowest, n + 1U);  // succesor sentinel
        return contributionsNonDominated(points);
    }
    points.emplace_back(refX, lowest, n + 1U);  // succesor sentinel
    return contributionsDominated(points);
}

template <typename T>
auto contributionsNonDominated(const std::vector<IndexedPoint2D<T>> &points) {
    std::size_t n = points.size() - 2U;
    auto contribution = new T[n]{0.0};
    py::capsule freeContributionsMemory(contribution, [](void *ptr) {
        std::unique_ptr<T[]>(static_cast<decltype(contribution)>(ptr));
    });
    const auto retval = py::array_t<T>({n},          // shape
                                       {sizeof(T)},  // stride
                                       contribution,
                                       freeContributionsMemory);
    // Process all the points
    for (std::size_t i = 1U, m = n + 1U; i < m; ++i) {
        const auto [pX, pY, index] = points[i];
        const auto [leftX, leftY, leftIndex] = points[i - 1U];
        const auto [rightX, rightY, rightIndex] = points[i + 1U];
        contribution[index] = (rightX - pX) * (leftY - pY);
    }
    // Ensure sentinel contribs are "infinite".
    constexpr auto max = std::numeric_limits<T>::max();
    contribution[points[0U].index] = max;
    contribution[points[n - 1U].index] = max;
    return retval;
}

#endif  // ANGUILLA_HYPERVOLUME_HVC2D_NG_HPP
