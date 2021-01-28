#pragma once

#ifndef ANGUILLA_DOMINATION_HPP
#define ANGUILLA_DOMINATION_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// STL
#include <vector>

// Anguilla
#include <anguilla/common/common.hpp>

/* References:

[2002:nsga-ii]
K. Deb, A. Pratap, S. Agarwal, & T. Meyarivan (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II
IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
*/

namespace dominance {
static constexpr const char *docstring = R"_(
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
[[nodiscard]] auto nonDominatedSort(const py::array_t<T> &_points, const std::optional<std::size_t> &maxRank) {
    static_assert(std::is_floating_point<T>::value,
                  "fast_non_dominated_sort is not meant to be instantiated with a non floating point type.");
    const auto points = _points.template unchecked<2>();
    assert(points.shape(0) >= 0);

    const auto nPoints = static_cast<std::size_t>(points.shape(0));
    const auto nDimensions = static_cast<std::size_t>(points.shape(1));
    if (nPoints == 0U) {
        return py::make_tuple(py::array_t<T>(0U), 0U);
    }

    std::vector<std::size_t> ranks(nPoints, 0U);
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
                //p_all = true;
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

    return py::make_tuple(as_pyarray<decltype(ranks)>(std::move(ranks)), rank);
}
}  // namespace dominance
#endif  // ANGUILLA_DOMINATION_HPP
