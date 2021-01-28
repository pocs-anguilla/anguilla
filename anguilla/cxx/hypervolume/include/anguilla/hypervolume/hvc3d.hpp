#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HVC3D_HPP
#define ANGUILLA_HYPERVOLUME_HVC3D_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

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

/* REFERENCES

[2009:hypervolume-hv3d]
N. Beume, C. Fonseca, M. Lopez-Ibanez, L. Paquete, & J. Vahrenhold (2009).
On the Complexity of Computing the Hypervolume Indicator
IEEE transactions on evolutionary computation, 13(5), 1075–1082.

[2011:hypervolume-3d]
M. Emmerich, & C. Fonseca (2011).
Computing Hypervolume Contributions in Low Dimensions: Asymptotically Optimal Algorithm and Complexity Results.
In Evolutionary Multi-Criterion Optimization (pp. 121–135). Springer Berlin Heidelberg.

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/Jtm08, https://git.io/JImE2

[2020:hypervolume]
A. P. Guerreiro, C. M. Fonseca, & L. Paquete. (2020).
The Hypervolume Indicator: Problems and Algorithms.
*/

/* Public interface */
namespace hvc3d {
static constexpr const char *docstring = R"_(
    Compute the hypervolume contributions for a set of 3-D points.

    Parameters
    ----------
    points
        The set of points. If the point set is not mutually non-dominated, \
        then dominated points are ignored.
    reference
        The reference point. Otherwise computed as the component-wise
        maximum over all the points.

    Returns
    -------
    np.ndarray
        The hypervolume contribution of each point, respectively.

    Notes
    -----
    Implements the HVC3D algorithm (see p. 22 of :cite:`2020:hypervolume`) \
    presented by :cite:`2011-hypervolume-3d` for computing AllContributions. \
    The implementation differs from the presentation of the reference paper \
    in that it assumes a minimization problem (instead of maximization). \
    It also incorporates some implementation details taken from \
    :cite:`2008:shark`. See also HV3D in [2011:hypervolume-3d].)_";

template <typename T, class Map>
auto contributions(const py::array_t<T> &_points, const std::optional<py::array_t<T>> &reference);

template <typename T>
struct MapValue {
    MapValue(T pY, std::size_t index) : pY(pY), index(index) {}

    T pY;
    std::size_t index;
};

#ifdef __cpp_lib_memory_resource
template <typename T>
using MapPair = std::pair<const T, MapValue<T>>;

template <typename T>
using MapAllocator = std::pmr::polymorphic_allocator<MapPair<T>>;

template <typename T>
using RBTreeMap = std::map<T, MapValue<T>, std::less<T>, MapAllocator<T>>;

template <typename T>
using BTreeMap = btree::map<T, MapValue<T>, std::less<T>, MapAllocator<T>, 1024>;

#else

template <typename T>
using RBTreeMap = std::map<T, MapValue<T>>;

template <typename T>
using BTreeMap = btree::map<T, MapValue<T>>;
#endif
}  // namespace hvc3d

/* Internal interface */
namespace __hvc3d {
template <typename T>
struct IndexedPoint3D;

template <typename T, class Map>
[[nodiscard]] auto contributions(const std::vector<IndexedPoint3D<T>> &points, const T refX, const T refY, const T refZ);

template <typename T>
struct Box3D {
    Box3D(T lX, T lY, T lZ, T uX, T uY, T uZ) : lX(lX), lY(lY), lZ(lZ), uX(uX), uY(uY), uZ(uZ) {}

    inline T volume() const {
        return (uX - lX) * (uY - lY) * (uZ - lZ);
    }

    T lX;
    T lY;
    T lZ;
    T uX;
    T uY;
    T uZ;
};

template <typename T>
struct IndexedPoint3D {
    IndexedPoint3D(T pX, T pY, T pZ, std::size_t index) : pX(pX), pY(pY), pZ(pZ), index(index) {}

    T pX;
    T pY;
    T pZ;
    std::size_t index;
};

template <typename T>
struct DominatedData {
    DominatedData(T pX, T pY, std::size_t index) : pX(pX), pY(pY), index(index) {}

    T pX;
    T pY;
    std::size_t index;
};

template <typename T>
using MapValue = hvc3d::MapValue<T>;
}  // namespace __hvc3d

/* Implementation of public interface. */
namespace hvc3d {
template <typename T, class Map>
[[nodiscard]] auto contributions(const py::array_t<T> &_points, const std::optional<py::array_t<T>> &_reference) {
    static_assert(std::is_floating_point<T>::value,
                  "HVC3D is not meant to be instantiated with a non floating point type.");

    const auto pointsR = _points.template unchecked<2>();
    assert(pointsR.shape(0) >= 0);
    const auto n = static_cast<std::size_t>(pointsR.shape(0));
    if (n == 0U) {
        return py::array_t<T>(0U);
    }

    constexpr auto lowest = std::numeric_limits<T>::lowest();
    T refX = lowest;
    T refY = lowest;
    T refZ = lowest;
    const bool refGiven = _reference.has_value();

    if (refGiven) {
        const auto referenceR = _reference.value().template unchecked<1>();
        refX = referenceR(0);
        refY = referenceR(1);
        refZ = referenceR(2);
    }

    std::vector<__hvc3d::IndexedPoint3D<T>> points;
    points.reserve(n + 1U);
    for (std::size_t i = 0U; i < n; ++i) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        const auto pZ = pointsR(i, 2);
        points.emplace_back(pX, pY, pZ, i);
        if (!refGiven) {
            refX = std::max(refX, pX);
            refY = std::max(refY, pY);
            refZ = std::max(refZ, pZ);
        }
    }
    std::sort(points.begin(), points.end(),
              [](auto const &l, auto const &r) { return l.pZ < r.pZ; });
    // Add dummy point that will close the boxes, as in the paper.
    points.emplace_back(lowest, lowest, refZ, n);

    return __hvc3d::contributions<T, Map>(points, refX, refY, refZ);
}
}  // namespace hvc3d

/* Internal implementations. */
namespace __hvc3d {
template <typename T, class Map>
auto contributions(const std::vector<IndexedPoint3D<T>> &points, const T refX, const T refY, const T refZ) {
    // Note: assumes the points are received sorted in ascending order by z-component.

#ifdef NDEBUG
    static_assert(std::numeric_limits<T>::has_quiet_NaN, "No quiet NaN.");
    constexpr auto NaN = std::numeric_limits<T>::quiet_NaN();
#else
    static_assert(std::numeric_limits<T>::has_signaling_NaN, "No signaling NaN.");
    constexpr auto NaN = std::numeric_limits<T>::signaling_NaN();
#endif

    // Here we allocate memory for the contributions, initialized to zero,
    // plus an additional element for both sentinel nodes.
    const std::size_t n = points.size();
    auto contribution = new T[n + 2U]{0.0};
    contribution[n - 1U] = NaN;
    contribution[n] = NaN;
    contribution[n + 1U] = NaN;

    py::capsule freeContributionsMemory(contribution, [](void *ptr) {
        std::unique_ptr<T[]>(static_cast<decltype(contribution)>(ptr));
    });

    // We work on a vector but will return a Numpy array.
    // It uses the vector's memory, which will be freed once the array
    // goes out of scope (handled by the py::capsule).
    const auto output = py::array_t<T>({n - 1U},     // shape
                                       {sizeof(T)},  // stride
                                       contribution,
                                       freeContributionsMemory);

// Create the sweeping structure.
#ifdef __cpp_lib_memory_resource
    std::unique_ptr<std::pmr::monotonic_buffer_resource> frontPoolPtr(new std::pmr::monotonic_buffer_resource(sizeof(hvc3d::MapPair<T>) * (n + 2U)));
    std::unique_ptr<Map> frontPtr(new Map(frontPoolPtr.get()));
    auto &front = *frontPtr;
#else
    Map front;
#endif

    // Create the lists of boxes.
    std::vector<std::deque<Box3D<T>>> boxLists(n + 2U);

    // A working buffer for tracking dominated nodes.
    std::vector<DominatedData<T>> dominated;

    {
        const auto [pX, pY, pZ, index] = points[0U];
        boxLists[index].emplace_front(pX, pY, pZ, refX, refY, NaN);
        constexpr auto lowest = std::numeric_limits<T>::lowest();
        // Add first point.
        front.try_emplace(pX, pY, index);
        // Add the sentinels.
        front.try_emplace(lowest, refY, n);
        front.try_emplace(refX, lowest, n + 1U);
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
            continue;  // 'p' is dominated by 'q'
        }
        if (!(nodeLeft->first < pX) && nodeLeft != front.begin()) {  // qX == pX
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
            auto &boxes = boxLists[leftIndex];
            while (!boxes.empty()) {
                auto &box = boxes.back();
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
        if (nodeRight != front.end()) {
            T volume = 0.0;
            const T rightX_0 = nodeRight->first;
            T rightX = rightX_0;
            const auto [rightY, rightIndex] = nodeRight->second;
            auto &boxes = boxLists[rightIndex];
            while (!boxes.empty()) {
                auto &box = boxes.front();
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
                boxLists[rightIndex].emplace_front(rightX_0, rightY, pZ, rightX, pY, NaN);
            }
            contribution[rightIndex] += volume;
        }

        // (c) Process the dominated points.
        {
            const auto leftY = nodeLeft->second.pY;
            T rightX = nodeRight->first;
            // note: process dominated indices in reverse order
            for (auto dIt = dominated.rbegin(), end = dominated.rend(); dIt != end; ++dIt) {
                const auto [dX, dY, dIndex] = *dIt;
                auto &boxes = boxLists[dIndex];
                // close boxes of dominated point 'd'
                for (auto &box : boxes) {
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

    return output;
}
}  // namespace __hvc3d

#endif  // ANGUILLA_HYPERVOLUME_HVC3D_HPP
