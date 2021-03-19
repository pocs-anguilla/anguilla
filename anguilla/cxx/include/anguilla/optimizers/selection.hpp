#pragma once

#ifndef ANGUILLA_OPTIMIZERS_SELECTION_HPP_
#define ANGUILLA_OPTIMIZERS_SELECTION_HPP_

// STL
#include <numeric>
#include <vector>

// Xtensor
#include <xtensor/xbuilder.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xsort.hpp>

// Anguilla
#include <anguilla/common/common.hpp>
#include <anguilla/hypervolume/hypervolume.hpp>
namespace ag = anguilla;

namespace anguilla {
namespace hypervolume {
template <typename T>
auto leastContributors(const xt::xtensor<T, 2U>& points, std::size_t k)
    -> std::vector<std::size_t> {
    std::vector<std::size_t> indices;
    indices.reserve(k);
    std::vector<std::size_t> activeIndices(points.shape(0U));
    std::iota(activeIndices.begin(), activeIndices.end(), 0U);
    for (auto i = 0U; i < k; ++i) {
        const auto activePoints =
            xt::view(points, xt::keep(activeIndices), xt::all());
        const auto contribs = contributions<T>(activePoints);
        std::size_t index = xt::argsort(contribs)[0];
        indices.push_back(activeIndices[index]);
        activeIndices.erase(std::next(activeIndices.begin(), index));
    }
    return indices;
}

template <typename T>
auto selection(const xt::xtensor<T, 2U>& points,
               const xt::xtensor<std::size_t, 1U>& ranks,
               const std::size_t targetSize) -> xt::xtensor<bool, 1U> {
    xt::xtensor<bool, 1U> selected = xt::zeros<bool>({points.shape(0U)});
    int nPendingSelect = (int)targetSize;
    std::size_t rank = 1U;
    while (nPendingSelect > 0) {
        // xt::from_indices returns a 2D tensor when the front has size > 1
        // so we need to use xt::squeeze
        const auto front =
            xt::squeeze(xt::from_indices(xt::argwhere(xt::equal(ranks, rank))));
        const int diff = nPendingSelect - (int)front.size();
        if (diff > 0) {
            // We must select all individuals in the current front.
            xt::index_view(selected, front) = true;
            nPendingSelect = diff;
            rank++;
        } else {
            // If diff == 0
            // Then all pending individuals are exactly in this front.
            xt::index_view(selected, front) = true;
            nPendingSelect = 0;
            // Otherwise
            // We select the rest of the pending individuals among individuals
            // in the current front by discarding the least contributors.
            // See also Lemma 1 in page 11 of [2007:mo-cma-es].
            if (diff < 0) {
                auto lc = leastContributors<T>(
                    xt::view(points, xt::keep(front), xt::all()),
                    (std::size_t)-diff);
                auto front_lc = xt::index_view(front, lc);
                xt::index_view(selected, front_lc) = false;
            }
        }
    }
    return selected;
};
} // namespace hypervolume
} // namespace anguilla

#endif // ANGUILLA_OPTIMIZERS_SELECTION_HPP_
