#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVC2D_HPP
#define ANGUILLA_HYPERVOLUME_HVC2D_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL
#include <algorithm>
#include <deque>
#include <numeric>
#include <vector>

/* REFERENCES

[2007:mo-cma-es]
C. Igel, N. Hansen, & S. Roth (2007).
Covariance Matrix Adaptation for Multi-objective OptimizationEvolutionary Computation, 15(1), 1–28.
NOTE: see the algorithm described in Lemma 1 of the paper, on page 11. 

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm0o

[2011:hypervolume-3d]
M. Emmerich, & C. Fonseca (2011).
Computing Hypervolume Contributions in Low Dimensions: Asymptotically Optimal Algorithm and Complexity Results.
In Evolutionary Multi-Criterion Optimization (pp. 121–135). Springer Berlin Heidelberg.

[2020:hypervolume]
A. P. Guerreiro, C. M. Fonseca, & L. Paquete. (2020).
The Hypervolume Indicator: Problems and Algorithms.
*/

/* Public interface */
namespace hvc2d {
static constexpr const char *docstring = R"_(
    Compute the exact hypervolume contributions for a set of 2-D points.

    Parameters
    ----------
    points
        The point set of points.
    reference: optional
        The reference point. Otherwise the extremum points are used to \
        compute the other points' contribution and are excluded, \
        defaulting to the maximum positive value.
    non_dominated
        By default we assume that points are non-dominated and use the faster \
        version that implements the algorithm from Lemma 1 described in \
        :cite:`2007:mo-cma-es`.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    The algorithm described in in Lemma 1 of :cite:`2007:mo-cma-es` \
    assumes mutually non-dominated points. The implementation uses this \
    algorithm by default. If some points may be dominated, an alternative \
    algorithm is provided, based on that by [2011:hypervolume-3d]. \
    This version also incorporates some aspects described in \
    :cite:`2007:mo-cma-es` and implemented in :cite:`2008:shark`. \
    In particular, the handling of extremum points.)_";

template <typename T>
[[nodiscard]] auto contributions(const py::array_t<T> &_points, const bool nonDominated);

template <typename T>
[[nodiscard]] auto contributionsWithRef(const py::array_t<T> &_points, const py::array_t<T> &_reference, const bool nonDominated);
}  // namespace hvc2d

/* Implementation internal interface */
namespace __hvc2d {
template <typename T>
struct IndexedPoint2D;

template <typename T>
[[nodiscard]] auto contributionsDominated(const std::vector<IndexedPoint2D<T>> &points, const bool noRef);

template <typename T>
[[nodiscard]] auto contributionsNonDominated(const std::vector<IndexedPoint2D<T>> &points, const bool noRef);

template <typename T>
[[nodiscard]] inline auto dummy(const std::size_t n);

template <typename T>
struct IndexedPoint2D {
    IndexedPoint2D(T pX, T pY, std::size_t index) : pX(pX), pY(pY), index(index) {}

    T pX;
    T pY;
    std::size_t index;
};

template <typename T>
using DominatedData = IndexedPoint2D<T>;

template <typename T>
struct Box2D {
    Box2D(T lX, T lY, T uX, T uY) : lX(lX), lY(lY), uX(uX), uY(uY) {}

    inline T volume() const {
        return (uX - lX) * (uY - lY);
    }

    T lX;
    T lY;
    T uX;
    T uY;
};
}  // namespace __hvc2d

/* Implementation of public interface. */
namespace hvc2d {
template <typename T>
auto contributions(const py::array_t<T> &_points, const bool nonDominated) {
    static_assert(std::is_floating_point<T>::value,
                  "HVC2D is not meant to be instantiated with a non floating point type.");
    const auto pointsR = _points.template unchecked<2>();
    assert(pointsR.shape(0) >= 0);
    const auto n = static_cast<std::size_t>(pointsR.shape(0));
    if (n == 0U) {
        return py::array_t<T>(0U);
    }
    if (n < 3U) {
        return __hvc2d::dummy<T>(n);
    }
    std::vector<__hvc2d::IndexedPoint2D<T>> points;
    points.reserve(n);
    for (auto i = 0U; i < n; i++) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        points.emplace_back(pX, pY, i);
    }
    std::sort(points.begin(), points.end(),
              [](auto const &l, auto const &r) { return l.pX < r.pX; });
    constexpr auto noRef = true;
    if (nonDominated) {
        return __hvc2d::contributionsNonDominated<T>(points, noRef);
    }
    return __hvc2d::contributionsDominated<T>(points, noRef);
}

template <typename T>
auto contributionsWithRef(const py::array_t<T> &_points, const py::array_t<T> &_reference, const bool nonDominated) {
    static_assert(std::is_floating_point<T>::value,
                  "HVC2D is not meant to be instantiated with a non floating point type.");
    const auto pointsR = _points.template unchecked<2>();
    assert(pointsR.shape(0) >= 0);
    const auto n = static_cast<size_t>(pointsR.shape(0));
    if (n == 0U) {
        return py::array_t<T>(0U);
    }
    constexpr auto lowest = std::numeric_limits<T>::lowest();
    const auto referenceR = _reference.template unchecked<1>();
    const auto refX = referenceR(0);
    const auto refY = referenceR(1);
    std::vector<__hvc2d::IndexedPoint2D<T>> points;
    points.reserve(n + 2U);
    points.emplace_back(lowest, refY, n);  // predecesor sentinel
    for (auto i = 0U; i < n; i++) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        points.emplace_back(pX, pY, i);
    }
    std::sort(std::next(points.begin()), points.end(),
              [](auto const &l, auto const &r) { return l.pX < r.pX; });
    points.emplace_back(refX, lowest, n + 1U);  // succesor sentinel
    constexpr auto noRef = false;
    if (nonDominated) {
        return __hvc2d::contributionsNonDominated<T>(points, noRef);
    }
    return __hvc2d::contributionsDominated<T>(points, noRef);
}
}  // namespace hvc2d

/* Internal implementations. */
namespace __hvc2d {
template <typename T>
auto contributionsNonDominated(const std::vector<IndexedPoint2D<T>> &points, const bool noRef) {
    // By convention, we assume two sentinel points.
    const auto n = points.size() - 2U;
    auto contribution = new T[n + 2U]{0.0};

    py::capsule freeContributionsMemory(contribution, [](void *ptr) {
        std::unique_ptr<T[]>(static_cast<decltype(contribution)>(ptr));
    });

    const auto retval = py::array_t<T>({noRef ? n + 2U : n},  // shape
                                       {sizeof(T)},           // stride
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
    const auto xSentinel = points.size() - 1U;
    const auto ySentinel = 0U;
    constexpr auto inf = std::numeric_limits<T>::max();
    contribution[points[xSentinel].index] = inf;
    contribution[points[ySentinel].index] = inf;

    return retval;
}

template <typename T>
auto contributionsDominated(const std::vector<IndexedPoint2D<T>> &points, const bool noRef) {
    // By convention, we assume two sentinel points.
    const auto n = points.size() - 2U;
    auto contribution = new T[n + 2U]{0.0};

    py::capsule freeContributionsMemory(contribution, [](void *ptr) {
        std::unique_ptr<T[]>(static_cast<decltype(contribution)>(ptr));
    });

    // We work on a vector but will return a Numpy array.
    // It uses the vector's memory, which will be freed once the array
    // goes out of scope (handled by the py::capsule).
    const auto retval = py::array_t<T>({noRef ? n + 2U : n},  // shape
                                       {sizeof(T)},           // stride
                                       contribution,
                                       freeContributionsMemory);

    // For keeping track of which points need processing.
    std::vector<bool> pending(n + 2U, true);

    // Create the lists of boxes.
    std::vector<std::deque<Box2D<T>>> boxLists(n + 2U);

    // A working buffer for tracking dominated nodes.
    std::vector<DominatedData<T>> dominated;

    // Process all the points.
    const auto sup = n + 2U;
    for (std::size_t i = 1U, m = n + 1U; i < m; ++i) {
        const auto [pX, pY, index] = points[i];
        auto [leftX, leftY, leftIndex] = points[i - 1U];
        if (!(leftY > pY)) {
            pending[i] = false;
            continue;  // 'p' is dominated by 'q'
        }
        if (!(leftX < pX)) {  // qX == pX
            dominated.emplace_back(leftX, leftY, leftIndex);
            pending[i - 1U] = false;
            leftX = points[i - 2U].pX;
            leftY = points[i - 2U].pY;
            leftIndex = points[i - 2U].index;
        }

        // Find dominated nodes.
        auto s = i + 1U;
        auto right = points[s];
        while (!(pY > right.pY)) {
            dominated.emplace_back(right.pX, right.pY, right.index);
            i = s;
            pending[s] = false;

            ++s;
            if (s == sup) break;
            right = points[s];
        }

        // Process left region.
        {
            T volume = 0.0;
            auto &boxes = boxLists[leftIndex];
            while (!boxes.empty()) {
                auto &box = boxes.back();
                if (pX < box.lX) {
                    // This box is dominated at this z-level
                    // so it can completed and added to the
                    // volume contribution of the left neighbour.
                    volume += box.volume();
                    boxes.pop_back();
                } else {
                    if (pX < box.uX) {
                        volume += box.volume();
                        // Modify box to reflect the dominance
                        // of the left neighbour in this part
                        // of the L region.
                        box.uX = pX;
                        // Stop removing boxes
                        break;
                    } else {
                        break;
                    }
                }
            }
            contribution[leftIndex] += volume;
        }

        // Process right region
        {
            T volume = 0.0;
            const T rightX_0 = right.pX;
            T rightX = rightX_0;
            auto &boxes = boxLists[right.index];
            while (!boxes.empty()) {
                auto &box = boxes.front();
                if (box.uY > pY) {
                    volume += box.volume();
                    rightX = box.uX;
                    boxes.pop_front();
                } else {
                    break;
                }
            }
            if (rightX > rightX_0) {
                boxLists[right.index].emplace_front(rightX_0, right.pY, rightX, pY);
            }
            contribution[right.index] += volume;
        }

        // Process the dominated points.
        {
            T rightX = right.pX;
            // note: process dominated indices in reverse order
            for (auto dIt = dominated.rbegin(), end = dominated.rend(); dIt != end; ++dIt) {
                const auto [dX, dY, dIndex] = *dIt;
                auto &boxes = boxLists[dIndex];
                // close boxes of dominated point 'd'
                for (auto &box : boxes) {
                    contribution[dIndex] += box.volume();
                }
                // open box for current point 'p'
                boxLists[index].emplace_front(dX, pY, rightX, dY);
                rightX = dX;
            }
            dominated.clear();
            boxLists[index].emplace_front(pX, pY, rightX, leftY);
        }
    }

    // Process pending boxes.
    for (std::size_t i = 1U, m = n + 1U; i < m; ++i) {
        if (pending[i]) {
            auto index = points[i].index;
            auto &boxes = boxLists[index];
            for (auto &box : boxes) {
                contribution[index] += box.volume();
            }
        }
    }

    // Ensure sentinel contribs are "infinite".
    const auto xSentinel = points.size() - 1U;
    const auto ySentinel = 0U;
    constexpr auto inf = std::numeric_limits<T>::max();
    contribution[points[xSentinel].index] = inf;
    contribution[points[ySentinel].index] = inf;

    return retval;
}

template <typename T>
auto dummy(const std::size_t n) {
    assert(n > 0U);
    constexpr auto inf = std::numeric_limits<T>::max();
    auto contribution = new T[n]{inf};

    py::capsule freeContributionsMemory(contribution, [](void *ptr) {
        std::unique_ptr<T[]>(static_cast<decltype(contribution)>(ptr));
    });

    // We work on a vector but will return a Numpy array.
    // It uses the vector's memory, which will be freed once the array
    // goes out of scope (handled by the py::capsule).
    const auto retval = py::array_t<T>({n},          // shape
                                       {sizeof(T)},  // stride
                                       contribution,
                                       freeContributionsMemory);

    return retval;
}

}  // namespace __hvc2d

#endif  // ANGUILLA_HYPERVOLUME_HVC2D_HPP
