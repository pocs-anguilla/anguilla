#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVC3D_HPP
#define ANGUILLA_HYPERVOLUME_HVC3D_HPP

// STL
#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef __cpp_lib_memory_resource
#include <memory_resource>
#endif

// PyBind11
#include <pybind11/stl.h>

// Xtensor
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

// Btree
#include <btree/map.h>

/* REFERENCES

[2007:mo-cma-es]
C. Igel, N. Hansen, & S. Roth (2007).
Covariance Matrix Adaptation for Multi-objective OptimizationEvolutionary
Computation, 15(1), 1–28. NOTE: see the algorithm described in Lemma 1 of the
paper, on page 11.

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm08, https://git.io/JImE2

[2009:hypervolume-hv3d]
N. Beume, C. Fonseca, M. Lopez-Ibanez, L. Paquete, & J. Vahrenhold (2009).
On the Complexity of Computing the Hypervolume Indicator
IEEE transactions on evolutionary computation, 13(5), 1075–1082.

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
namespace hvc3d {
/* Public interface */
static constexpr const char* docstring = R"_(
    Compute the hypervolume contributions for a set of 3-D points.

    Parameters
    ----------
    points
        The set of points. If the point set is not mutually non-dominated, \
        then dominated points are ignored.
    reference
        The reference point. Otherwise computed as the component-wise
        maximum over all the points.
    preferExtrema
        Whether to favor the extremum points by giving them infinite \
        contribution.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the HVC3D algorithm (see p. 22 of :cite:`2020:hypervolume`) \
    presented by :cite:`2011:hypervolume-3d` for computing AllContributions. \
    The implementation differs from the presentation of the reference paper \
    in that it assumes a minimization problem (instead of maximization). \
    It also incorporates some implementation details taken from \
    :cite:`2008:shark`. See also HV3D in [2011:hypervolume-3d].
    The implementation also incorporates some aspects described in \
    :cite:`2007:mo-cma-es` regarding the handling of extremum points.)_";

template <typename T, class Map>
auto contributions(const xt::xtensor<T, 2U>& inputPoints,
                   const std::optional<xt::xtensor<T, 1U>>& inputReference,
                   bool preferExtrema) -> xt::xtensor<T, 1U>;

template <typename T> struct MapValue {
    MapValue(T pY, std::size_t index) : pY(pY), index(index) {}

    T pY;
    std::size_t index;
};

template <typename T> using MapPair = std::pair<const T, MapValue<T>>;

#ifdef __cpp_lib_memory_resource

template <typename T>
using MapAllocator = std::pmr::polymorphic_allocator<MapPair<T>>;

template <typename T>
using RBTreeMap = std::map<T, MapValue<T>, std::less<T>, MapAllocator<T>>;

template <typename T>
using BTreeMap =
    btree::map<T, MapValue<T>, std::less<T>, MapAllocator<T>, 1024>;

#else

template <typename T> using RBTreeMap = std::map<T, MapValue<T>>;

template <typename T> using MapAllocator = std::allocator<MapPair<T>>;

template <typename T>
using BTreeMap =
    btree::map<T, MapValue<T>, std::less<T>, MapAllocator<T>, 1024>;
#endif

/* Internal interface */
namespace internal {
template <typename T> struct IndexedPoint3D;

template <typename T> struct ExtremaData;

template <typename T, class Map>
[[nodiscard]] auto contributions(const std::vector<IndexedPoint3D<T>>& points,
                                 const T refX, const T refY, const T refZ,
                                 const ExtremaData<T> extremaData)
    -> xt::xtensor<T, 1U>;

template <typename T> struct Box3D {
    Box3D(T lX, T lY, T lZ, T uX, T uY, T uZ)
        : lX(lX), lY(lY), lZ(lZ), uX(uX), uY(uY), uZ(uZ) {}

    inline T volume() const { return (uX - lX) * (uY - lY) * (uZ - lZ); }

    T lX;
    T lY;
    T lZ;
    T uX;
    T uY;
    T uZ;
};

template <typename T> struct IndexedPoint3D {
    IndexedPoint3D(T pX, T pY, T pZ, std::size_t index)
        : pX(pX), pY(pY), pZ(pZ), index(index) {}

    T pX;
    T pY;
    T pZ;
    std::size_t index;
};

template <typename T> struct DominatedData {
    DominatedData(T pX, T pY, std::size_t index)
        : pX(pX), pY(pY), index(index) {}

    T pX;
    T pY;
    std::size_t index;
};

template <typename T> using MapValue = hvc3d::MapValue<T>;

template <typename T> struct ExtremaData {
    std::size_t extA_i;
    std::size_t extB_i;
    std::size_t extC_i;
    bool preferExtrema;
};
} // namespace internal

/* Implementation of public interface. */
template <typename T, class Map>
[[nodiscard]] auto
contributions(const xt::xtensor<T, 2U>& inputPoints,
              const std::optional<xt::xtensor<T, 1U>>& inputReference,
              bool preferExtrema) -> xt::xtensor<T, 1U> {
    static_assert(std::is_floating_point<T>::value,
                  "HVC3D is not meant to be instantiated with a non floating "
                  "point type.");

    const std::size_t n = inputPoints.shape(0U);
    if (n == 0U) {
        return xt::zeros<T>({0U});
    }

    constexpr auto lowest = std::numeric_limits<T>::lowest();
    T refX = lowest;
    T refY = lowest;
    T refZ = lowest;
    const bool refGiven = inputReference.has_value();

    if (refGiven) {
        refX = (*inputReference)[0U];
        refY = (*inputReference)[1U];
        refZ = (*inputReference)[2U];
    }

    std::vector<internal::IndexedPoint3D<T>> points;
    points.reserve(n);

    // To determine the extremum points we need to keep track
    // of the three best extrema.

    constexpr auto max = std::numeric_limits<T>::max();
    T extA_x = max;
    T extA_y = max;
    std::size_t extA_i = n;

    T extB_x = max;
    T extB_z = max;
    std::size_t extB_i = n;

    T extC_y = max;
    T extC_z = max;
    std::size_t extC_i = n;

    for (std::size_t i = 0U; i < n; ++i) {
        const auto pX = inputPoints(i, 0U);
        const auto pY = inputPoints(i, 1U);
        const auto pZ = inputPoints(i, 2U);
        points.emplace_back(pX, pY, pZ, i);
        if (preferExtrema) {
            if (pX < extA_x && pY < extA_y) {
                extA_x = pX;
                extA_y = pY;
                extA_i = i;
            }
            if (pX < extB_x && pZ < extB_z) {
                extB_x = pX;
                extB_z = pZ;
                extB_i = i;
            }
            if (pY < extC_y && pZ < extC_z) {
                extC_y = pY;
                extC_z = pZ;
                extC_i = i;
            }
        }
        if (!refGiven) {
            refX = std::max(refX, pX);
            refY = std::max(refY, pY);
            refZ = std::max(refZ, pZ);
        }
    }

    internal::ExtremaData<T> extremaData;
    extremaData.extA_i = extA_i;
    extremaData.extB_i = extB_i;
    extremaData.extC_i = extC_i;
    extremaData.preferExtrema = preferExtrema;

    std::sort(points.begin(), points.end(),
              [](auto const& l, auto const& r) { return l.pZ < r.pZ; });

    return internal::contributions<T, Map>(points, refX, refY, refZ,
                                           extremaData);
}

/* Internal implementations. */
namespace internal {
template <typename T, class Map>
auto contributions(const std::vector<IndexedPoint3D<T>>& points, const T refX,
                   const T refY, const T refZ, const ExtremaData<T> extremaData)
    -> xt::xtensor<T, 1U> {
    // Note: assumes the points are received sorted in ascending order by
    // z-component.

#ifdef NDEBUG
    static_assert(std::numeric_limits<T>::has_quiet_NaN, "No quiet NaN.");
    constexpr auto NaN = std::numeric_limits<T>::quiet_NaN();
#else
    static_assert(std::numeric_limits<T>::has_signaling_NaN,
                  "No signaling NaN.");
    constexpr auto NaN = std::numeric_limits<T>::signaling_NaN();
#endif

    // Here we allocate memory for the contributions, initialized to zero,
    // plus an additional element for both sentinel nodes.
    const std::size_t n = points.size();
    xt::xtensor<T, 1U> contribution = xt::zeros<T>({n + 1U});
    contribution[n] = NaN;

// Create the sweeping structure.
#ifdef __cpp_lib_memory_resource
    std::unique_ptr<std::pmr::monotonic_buffer_resource> frontPoolPtr(
        new std::pmr::monotonic_buffer_resource(sizeof(hvc3d::MapPair<T>) *
                                                (n + 2U)));
    std::unique_ptr<Map> frontPtr(new Map(frontPoolPtr.get()));
    auto& front = *frontPtr;
#else
    Map front;
#endif

    // Create the lists of boxes.
    std::vector<std::deque<Box3D<T>>> boxLists(n + 1U);

    // A working buffer for tracking dominated nodes.
    std::vector<DominatedData<T>> dominated;

    {
        constexpr auto lowest = std::numeric_limits<T>::lowest();
        // Add the sentinels.
        front.try_emplace(lowest, refY, n);
        front.try_emplace(refX, lowest, n);
        // Process the first point.
        const auto [pX, pY, pZ, index] = points[0U];
        boxLists[index].emplace_front(pX, pY, pZ, refX, refY, NaN);
        front.try_emplace(pX, pY, index);
    }

    // Process all the points.
    for (std::size_t i = 1U; i < n; ++i) {
        const auto [pX, pY, pZ, index] = points[i];

        // Note:
        //  - 'left' is also called 'q' in the papers
        //  - 'right' is also called 't' (or 's')

        // We seek the greatest 'qX' s.t. 'qX' =< 'pX'.
        // The interface of lower_bound is s.t. it returns an iterator that
        // points to the first 'qX' s.t. 'qX' >= 'pX'.
        // Otherwise, the iterator points to the end.
        // Sentinels guarantee that the following call succeds.
        auto nodeLeft = front.lower_bound(pX);
        assert(nodeLeft != front.end());
        assert(!(nodeLeft->first < pX));
        if ((nodeLeft->first > pX) && nodeLeft != front.begin()) {
            --nodeLeft;
        }
        assert(!(nodeLeft->first > pX));
        // Find if 'p' is dominated by 'q' or otherwise.
        if (!(nodeLeft->second.pY > pY)) {
            continue; // 'p' is dominated by 'q'
        }
        if (!(nodeLeft->first < pX) && nodeLeft != front.begin()) { // qX == pX
            --nodeLeft;
        }

        // (a) Find all points dominated by p, remove them
        // from the sweeping structure and add them to
        // the list of dominated nodes.
        // Also here we find the lowest t_x, such that p_x < t_x.
        auto nodeRight = nodeLeft;
        ++nodeRight;
        while ((nodeRight != front.end()) && !(pY > nodeRight->second.pY)) {
            const auto [pY, index] = nodeRight->second;
            dominated.emplace_back(nodeRight->first, pY, index);
            ++nodeRight;
        }

        // (b) Process "left" region (symmetric to the paper).
        {
            const auto leftIndex = (nodeLeft->second).index;
            T volume = 0.0;
            auto& boxes = boxLists[leftIndex];
            while (!boxes.empty()) {
                auto& box = boxes.back();
                if (pX < box.lX) {
                    // This box is dominated at this z-level
                    // so it can completed and added to the
                    // volume contribution of the left neighbour.
                    box.uZ = pZ;
                    volume += box.volume();
                    boxes.pop_back();
                } else {
                    if (!(pX > box.uX)) {
                        box.uZ = pZ;
                        volume += box.volume();
                        // Modify box to reflect the dominance
                        // of the left neighbour in this part
                        // of the L region.
                        box.uX = pX;
                        box.uZ = NaN;
                        box.lZ = pZ;
                        // Stop removing boxes
                        break;
                    } else {
                        break;
                    }
                }
            }
            contribution[leftIndex] += volume;
        }

        // (d) Process "right" region (symmetric to the paper).
        {
            T volume = 0.0;
            const T rightX_0 = nodeRight->first;
            T rightX = rightX_0;
            const auto [rightY, rightIndex] = nodeRight->second;
            auto& boxes = boxLists[rightIndex];
            while (!boxes.empty()) {
                auto& box = boxes.front();
                if (!(box.uY < pY)) {
                    box.uZ = pZ;
                    volume += box.volume();
                    rightX = box.uX;
                    boxes.pop_front();
                } else {
                    break;
                }
            }
            if (!(rightX < rightX_0)) {
                boxLists[rightIndex].emplace_front(rightX_0, rightY, pZ, rightX,
                                                   pY, NaN);
            }
            contribution[rightIndex] += volume;
        }

        // (c) Process the dominated points.
        {
            const auto leftY = nodeLeft->second.pY;
            T rightX = nodeRight->first;
            // note: process dominated indices in reverse order
            for (auto dIt = dominated.rbegin(), end = dominated.rend();
                 dIt != end; ++dIt) {
                const auto [dX, dY, dIndex] = *dIt;
                auto& boxes = boxLists[dIndex];
                // close boxes of dominated point 'd'
                for (auto& box : boxes) {
                    box.uZ = pZ;
                    contribution[dIndex] += box.volume();
                }
                // open box for current point 'p'
                boxLists[index].emplace_front(dX, pY, pZ, rightX, dY, NaN);
                rightX = dX;
            }
            dominated.clear();
            boxLists[index].emplace_front(pX, pY, pZ, rightX, leftY, NaN);
        }
        auto hintNode = front.erase(std::next(nodeLeft), nodeRight);

        // (e) insert point
        front.try_emplace(hintNode, pX, pY, index);
    }

    // Close boxes of remaining points, as in Shark.
    for (auto it = front.begin(), end = front.end(); it != end; ++it) {
        const auto index = it->second.index;
        auto& boxes = boxLists[index];
        for (auto& box : boxes) {
            box.uZ = refZ;
            contribution[index] += box.volume();
        }
    }

    // Optionally, give preference to extrema.
    if (extremaData.preferExtrema) {
        constexpr auto inf = std::numeric_limits<T>::max();
        contribution[extremaData.extA_i] = inf;
        contribution[extremaData.extB_i] = inf;
        contribution[extremaData.extC_i] = inf;
    }

    return xt::view(contribution, xt::range(0, n));
}
} // namespace internal
} // namespace hvc3d
} // namespace anguilla

#endif // ANGUILLA_HYPERVOLUME_HVC3D_HPP
