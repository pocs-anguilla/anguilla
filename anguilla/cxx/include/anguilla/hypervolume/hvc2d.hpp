#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVC2D_HPP
#define ANGUILLA_HYPERVOLUME_HVC2D_HPP

// STL
#include <algorithm>
#include <deque>
#include <numeric>
#include <vector>

// Xtensor
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>

/* REFERENCES

[2007:mo-cma-es]
C. Igel, N. Hansen, & S. Roth (2007).
Covariance Matrix Adaptation for Multi-objective OptimizationEvolutionary
Computation, 15(1), 1–28. NOTE: see the algorithm described in Lemma 1 of the
paper, on page 11.

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm0o

[2011:hypervolume-3d]
M. Emmerich, & C. Fonseca (2011).
Computing Hypervolume Contributions in Low Dimensions: Asymptotically Optimal
Algorithm and Complexity Results. In Evolutionary Multi-Criterion Optimization
(pp. 121–135). Springer Berlin Heidelberg.

[2020:hypervolume]
A. P. Guerreiro, C. M. Fonseca, & L. Paquete. (2020).
The Hypervolume Indicator: Problems and Algorithms.
*/

namespace anguilla {
namespace hvc2d {
/* Public interface */
static constexpr const char* docstring = R"_(
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
[[nodiscard]] auto contributions(const xt::xtensor<T, 2U>& inputPoints)
    -> xt::xtensor<T, 1U>;

template <typename T>
[[nodiscard]] auto
contributionsWithRef(const xt::xtensor<T, 2U>& inputPoints,
                     const xt::xtensor<T, 1U>& inputReference)
    -> xt::xtensor<T, 1U>;

/* Implementation internal interface */
namespace internal {
template <typename T> struct IndexedPoint2D {
    IndexedPoint2D(T pX, T pY, std::size_t index)
        : pX(pX), pY(pY), index(index) {}

    T pX;
    T pY;
    std::size_t index;
};

template <typename T>
[[nodiscard]] auto
contributionsDominated(const std::vector<IndexedPoint2D<T>>& points,
                       const bool noRef) -> xt::xtensor<T, 1U>;

template <typename T>
[[nodiscard]] auto
contributionsNonDominated(const std::vector<IndexedPoint2D<T>>& points,
                          const bool noRef) -> xt::xtensor<T, 1U>;

} // namespace internal

/* Implementation of public interface. */
template <typename T>
auto contributions(const xt::xtensor<T, 2U>& inputPoints)
    -> xt::xtensor<T, 1U> {
    static_assert(std::is_floating_point<T>::value,
                  "HVC2D is not meant to be instantiated with a non floating "
                  "point type.");
    const std::size_t n = inputPoints.shape(0U);
    if (n == 0U) {
        return xt::zeros<T>({0U});
    }
    if (n < 3U) {
        return xt::zeros<T>({n});
    }
    std::vector<internal::IndexedPoint2D<T>> points;
    points.reserve(n);
    for (auto i = 0U; i < n; i++) {
        const auto pX = inputPoints(i, 0);
        const auto pY = inputPoints(i, 1);
        points.emplace_back(pX, pY, i);
    }
    std::sort(points.begin(), points.end(),
              [](auto const& l, auto const& r) { return l.pX < r.pX; });
    constexpr auto noRef = true;
    return internal::contributionsNonDominated<T>(points, noRef);
}

template <typename T>
auto contributionsWithRef(const xt::xtensor<T, 2U>& inputPoints,
                          const xt::xtensor<T, 1U>& inputReference)
    -> xt::xtensor<T, 1U> {
    static_assert(std::is_floating_point<T>::value,
                  "HVC2D is not meant to be instantiated with a non floating "
                  "point type.");
    const std::size_t n = inputPoints.shape(0U);
    if (n == 0U) {
        return xt::zeros<T>({0U});
    }
    constexpr auto lowest = std::numeric_limits<T>::lowest();
    const auto refX = inputReference[0U];
    const auto refY = inputReference[1U];
    std::vector<internal::IndexedPoint2D<T>> points;
    points.reserve(n + 2U);
    points.emplace_back(lowest, refY, n); // predecesor sentinel
    for (auto i = 0U; i < n; i++) {
        const auto pX = inputPoints(i, 0);
        const auto pY = inputPoints(i, 1);
        points.emplace_back(pX, pY, i);
    }
    std::sort(std::next(points.begin()), points.end(),
              [](auto const& l, auto const& r) { return l.pX < r.pX; });
    points.emplace_back(refX, lowest, n + 1U); // succesor sentinel
    constexpr auto noRef = false;
    return internal::contributionsNonDominated<T>(points, noRef);
}

/* Internal implementations. */
namespace internal {
template <typename T>
auto contributionsNonDominated(const std::vector<IndexedPoint2D<T>>& points,
                               const bool noRef) -> xt::xtensor<T, 1U> {
    // By convention, we assume two sentinel points.
    const std::size_t n = points.size() - 2U;
    xt::xtensor<T, 1U> contribution = xt::zeros<T>({noRef ? n + 2U : n});

    // Process all the points
    for (std::size_t i = 1U, m = n + 1U; i < m; ++i) {
        const auto [pX, pY, index] = points[i];
        const auto leftY = points[i - 1U].pY;
        const auto rightX = points[i + 1U].pX;
        contribution[index] = (rightX - pX) * (leftY - pY);
    }

    // Ensure sentinel contribs are "infinite".
    if (noRef) {
        const auto xSentinel = points.size() - 1U;
        const auto ySentinel = 0U;
        constexpr auto inf = std::numeric_limits<T>::max();
        contribution[points[xSentinel].index] = inf;
        contribution[points[ySentinel].index] = inf;
    }

    return contribution;
}

} // namespace internal
} // namespace hvc2d
} // namespace anguilla

#endif // ANGUILLA_HYPERVOLUME_HVC2D_HPP
