#pragma once

#ifndef ANGUILLA_HYPERVOLUME_HV3D_HPP
#define ANGUILLA_HYPERVOLUME_HV3D_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// STL
#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#ifdef __cpp_lib_memory_resource
#include <memory_resource>
#endif

// Btree
#include <btree/map.h>

// Anguilla
#include <anguilla/common/common.hpp>

/* REFERENCES

[2009:hypervolume-hv3d]
N. Beume, C. Fonseca, M. Lopez-Ibanez, L. Paquete, & J. Vahrenhold (2009).
On the Complexity of Computing the Hypervolume Indicator
IEEE transactions on evolutionary computation, 13(5), 1075–1082.

[2008:shark]
C. Igel, V. Heidrich-Meisner, & T. Glasmachers (2008).
SharkJournal of Machine Learning Research, 9, 993–996.
URL: https://git.io/JImE2

[2020:hypervolume]
A. P. Guerreiro, C. M. Fonseca, & L. Paquete. (2020).
The Hypervolume Indicator: Problems and Algorithms.
*/

/* Public interface */
namespace hv3d {
static constexpr const char* docstring = R"_(
    Calculate the exact hypervolume indicator for a set of 3-D points

    Parameters
    ----------
    points
        The point set of points.
    reference: optional
        The reference point. Otherwise computed as the component-wise
        maximum over all the points.

    Returns
    -------
    float
        The hypervolume indicator.

    Notes
    -----
    Implements Algorithm 1 from :cite:`2009-hypervolume-hv3d`, \
    but with the difference of assuming a minimization problem. \
    See also Algorithm HV3D presented in :cite:`2020:hypervolume`, and the \
    explanation from sec. 4.1 of p. 6. of :cite:`2009-hypervolume-hv3d`.

    The differences in the implementation, w.r.t. the algorithm description \
    in the paper, are primarily due to them assuming a maximization \
    problem and the implementation the opposite (minimization).

    The following figure shows an example of how the algorithm is \
    transformed for working with a minimization problem. Note that \
    in both cases, the x-coordinates are assumed to be sorted in \
    ascending order.

    .. image:: /figures/hv3d_min.png
       :width: 750
       :alt: Example of the algorithm for minimization problem.)_";

/* Supporting datatypes for the implementation. */

#ifdef __cpp_lib_memory_resource
template <typename T> using MapPair = std::pair<const T, T>;

template <typename T>
using MapAllocator = std::pmr::polymorphic_allocator<MapPair<T>>;

template <typename T>
using RBTreeMap = std::map<T, T, std::less<T>, MapAllocator<T>>;

template <typename T>
using BTreeMap = btree::map<T, T, std::less<T>, MapAllocator<T>>;

#else

template <typename T> using RBTreeMap = std::map<T, T>;

template <typename T> using BTreeMap = btree::map<T, T>;
#endif

template <typename T, class Map = BTreeMap<T>>
[[nodiscard]] auto calculate(const py::array_t<T>& _points,
                             const std::optional<py::array_t<T>>& _reference,
                             bool ignoreDominated = false);

} // namespace hv3d

/* Internal interface */
namespace __hv3d {
template <typename T, class Map>
[[nodiscard]] auto calculate(const std::vector<Point3D<T>>& points,
                             const T refX, const T refY, const T refZ);
}

/* Implementation of public interface. */
namespace hv3d {
template <typename T, class Map>
auto calculate(const py::array_t<T>& _points,
               const std::optional<py::array_t<T>>& _reference,
               bool ignoreDominated) {
    static_assert(
        std::is_floating_point<T>::value,
        "HV3D is not meant to be instantiated with a non floating point type.");

    const auto pointsR = _points.template unchecked<2>();
    const auto n = static_cast<std::size_t>(pointsR.shape(0));

    if (n == 0U) {
        return 0.0;
    }

    constexpr auto lowest = std::numeric_limits<T>::lowest();
    T refX = lowest;
    T refY = lowest;
    T refZ = lowest;
    const bool refGiven = _reference.has_value();

    if (refGiven) {
        const auto referenceR = _reference->template unchecked<1>();
        refX = referenceR(0);
        refY = referenceR(1);
        refZ = referenceR(2);
    }

    std::vector<Point3D<T>> points;
    points.reserve(n);

    for (std::size_t i = 0U; i < n; ++i) {
        const auto pX = pointsR(i, 0);
        const auto pY = pointsR(i, 1);
        const auto pZ = pointsR(i, 2);

        if (refGiven && ignoreDominated) {
            if (pX < refX && pY < refY && pZ < refZ) {
                points.emplace_back(pX, pY, pZ);
            }
        } else {
            points.emplace_back(pX, pY, pZ);
        }

        if (!refGiven) {
            refX = std::max(refX, pX);
            refY = std::max(refY, pY);
            refZ = std::max(refZ, pZ);
        }
    }

    std::sort(points.begin(), points.end(),
              [](auto const& lhs, auto const& rhs) { return lhs.z < rhs.z; });

    return __hv3d::calculate<T, Map>(points, refX, refY, refZ);
}
} // namespace hv3d

namespace __hv3d {
template <typename T, class Map>
auto calculate(const std::vector<Point3D<T>>& points, const T refX,
               const T refY, const T refZ) {
    // Note: assumes the points are received sorted in ascending order by
    // z-component.

    // The algorithm works by performing sweeping in the z-axis,
    // and it uses a tree with balanced height as its sweeping structure
    // (e.g. an AVL tree or red-black tree) in which the keys are
    // the x-coordinates and the values are the y-coordinates.
    // See p. 6 of [2009:hypervolume-hv3d].

#ifdef __cpp_lib_memory_resource
    std::unique_ptr<std::pmr::monotonic_buffer_resource> frontPoolPtr(
        new std::pmr::monotonic_buffer_resource(sizeof(hv3d::MapPair<T>) *
                                                points.size()));
    std::unique_ptr<Map> frontPtr(new Map(frontPoolPtr.get()));
    auto& front = *frontPtr;
#else
    Map front;
#endif

    // As explained in [2009:hypervolume-hv3d], we use two sentinel points
    // to ease the handling of boundary cases by ensuring that succ(p_x)
    // and pred(p_x) are defined for any other p_x in the tree.
    constexpr auto lowest = std::numeric_limits<T>::lowest();
    front.try_emplace(lowest, refY);
    front.try_emplace(refX, lowest);

    // The first point from the set is added.
    const auto [pX, pY, pZ] = points[0U];
    T lastZ = pZ;
    front.try_emplace(pX, pY);

    T area = (refX - pX) * (refY - pY);
    T volume = 0.0;

    // and then the rest of the points are processed
    for (std::size_t i = 1U, n = points.size(); i < n; i++) {
        const auto [pX, pY, pZ] = points[i];

        // find greatest q_x, such that q_x <= p_x
        auto nodeQ = front.lower_bound(pX);
        assert(nodeQ != front.end());
        assert(!(nodeQ->first < pX));
        if ((nodeQ->first > pX) && nodeQ != front.begin()) {
            --nodeQ;
        }
        assert(!(nodeQ->first > pX));
        if (!(nodeQ->second > pY)) {
            continue; // 'p' is dominated by 'q'
        }
        if (!(nodeQ->first < pX) && nodeQ != front.begin()) { // qX == pX
            --nodeQ;
        }

        volume += area * (pZ - lastZ);
        lastZ = pZ;

        // remove dominated points and their area contributions
        auto nodeS = nodeQ;
        ++nodeS;
        T sX = nodeS->first;
        T sY = nodeS->second;
        T prevX = pX;
        T prevY = nodeQ->second;

        area -= (sX - prevX) * (refY - prevY);
        while ((nodeS != front.end()) && !(pY > nodeS->second)) {
            prevX = sX;
            prevY = sY;
            // 'nodeS' points to a dominated point before calling erase
            // and its successor afterwards.
            nodeS = front.erase(nodeS);
            sX = nodeS->first;
            sY = nodeS->second;
            area -= (sX - prevX) * (refY - prevY);
        }

        // add the new point (here 's' is 't' in the paper)
        area += (sX - pX) * (refY - pY);
        front.try_emplace(nodeS, pX, pY);
    }

    // last point's contribution to the volume
    volume += area * (refZ - lastZ);
    return volume;
}
} // namespace __hv3d

#endif // ANGUILLA_HYPERVOLUME_HV3D_HPP
