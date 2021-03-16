#pragma once

#ifndef ANGUILLA_DOMINATION_HPP
#define ANGUILLA_DOMINATION_HPP

// STL
#include <vector>

// Xtensor
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xbuilder.hpp>

// PyBind11
#include <pybind11/stl.h>
namespace py = pybind11;

// Anguilla
#include <anguilla/common/common.hpp>

/* References:

[2002:nsga-ii]
K. Deb, A. Pratap, S. Agarwal, & T. Meyarivan (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II
IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
*/

namespace anguilla {
namespace dominance {
static constexpr const char* NON_DOMINATED_SORT_DOCTSTRING = R"_(
    Compute the non-domination rank of given points.

    Parameters
    ----------
    points
        The set of points.
    max_rank: optional
        The maximum rank to sort up to.

    Returns
    -------
    Tuple[np.ndarray, int]
        The array with the non-domination ranks and the maximum rank.

    Notes
    -----
    Implements the algorithm presented in :cite:`2002:nsga-ii`.)_";

template <typename T>
[[nodiscard]] auto
nonDominatedSort(const xt::xtensor<T, 2U>& points,
                 const std::optional<std::size_t>& maxRank = std::nullopt)
    -> std::tuple<xt::xtensor<T, 1U>, std::size_t> {
    static_assert(std::is_floating_point<T>::value,
                  "nonDominatedSort is not meant to be instantiated "
                  "with a non floating point type.");

    const auto nPoints = points.shape(0U);
    const auto nDimensions = points.shape(1U);
    xt::xtensor<T, 1U> ranks = xt::zeros<std::size_t>({nPoints});
    if (nPoints == 0U) {
        return std::make_tuple(ranks, 0U);
    }
    // number of points that dominate each point
    std::vector<std::size_t> ns(nPoints, 0U);
    // dominated solutions of each point
    std::vector<std::vector<std::size_t>> ss;
    ss.reserve(nPoints);
    // the current front of non-dominated points
    std::vector<std::size_t> front;
    front.reserve(nPoints);
    // the next front of non-dominated points (Q in the paper)
    std::vector<std::size_t> nextFront;
    nextFront.reserve(nPoints);
    bool pAny;
    bool qAny;
    for (std::size_t i = 0U; i != nPoints; i++) {
        std::vector<std::size_t> s;
        s.reserve(nPoints - 1U);
        for (std::size_t j = 0U; j != nPoints; j++) {
            if (i != j) {
                pAny = false;
                qAny = false;
                for (std::size_t k = 0U; k != nDimensions; k++) {
                    // p[i] strictly dominates p[j] iff
                    // 1) p[i, k] <= p[j, k] for all k
                    // 2) p[i, k] < p[j, k] for any k
                    // 1 equiv) not (p[j, k] < p[i, k] for any k)
                    pAny = pAny || points(i, k) < points(j, k);
                    qAny = qAny || points(j, k) < points(i, k);
                }
                if (!qAny && pAny) {
                    s.emplace_back(j);
                } else {
                    if (!pAny && qAny) {
                        ns[i]++;
                    }
                }
            }
        }
        if (ns[i] == 0U) {
            ranks[i] = 1U;
            front.emplace_back(i);
        }
        ss.emplace_back(s);
    }
    std::size_t rank = 1U;
    while (!front.empty() && (!maxRank.has_value() || rank != maxRank)) {
        nextFront.clear();
        for (std::size_t i : front) {
            for (std::size_t j : ss[i]) {
                ns[j]--;
                if (ns[j] == 0U) {
                    ranks[j] = rank + 1U;
                    nextFront.emplace_back(j);
                }
            }
        }
        if (!nextFront.empty()) {
            rank++;
        }
        front.swap(nextFront);
    }

    return std::make_tuple(ranks, rank);
}

} // namespace dominance
} // namespace anguilla
#endif // ANGUILLA_DOMINATION_HPP
