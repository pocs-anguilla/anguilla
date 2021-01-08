#ifndef ANGUILLA_HYPERVOLUME_HV3D_HPP
#define ANGUILLA_HYPERVOLUME_HV3D_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>
namespace py = pybind11;

#include "btree/map.h"
#include "common.hpp"

namespace hv3d {

static constexpr const char *docstring =
    "Calculate the exact hypervolume indicator for a set of 3-D points.";

template <typename T = double>
using RBTreeMap = std::map<T, T>;

template <typename T = double>
using BTreeMap = btree::map<T, T>;

template <typename T = double, class Map = BTreeMap<T>>
T calculate(Point3DList<T> &points, const Point3D<T> &refPoint) {
    if (points.size() == 0U) {
        return 0.0;
    }

    std::sort(points.begin(), points.end(),
              [](auto const &lhs, auto const &rhs) { return lhs.z < rhs.z; });

    // The algorithm works by performing sweeping in the z-axis,
    // and it uses a tree with balanced height as its sweeping structure
    // (e.g. an AVL tree or red-black tree) in which the keys are
    // the x-coordinates and the values are the y-coordinates.
    // See p. 6 of [2009:hypervolume-hv3d].
    Map xyFront;

    // As explained in [2009:hypervolume-hv3d], we use two sentinel points
    // to ease the handling of boundary cases by ensuring that succ(p_x)
    // and pred(p_x) are defined for any other p_x in the tree.
    constexpr auto lowest = std::numeric_limits<T>::lowest();

    const T refX = refPoint.x;
    const T refY = refPoint.y;
    const T refZ = refPoint.z;

    xyFront[refX] = lowest;
    xyFront[lowest] = refY;

    // The first point from the set is added.
    T pX = points[0].x;
    T pY = points[0].y;
    T pZ = points[0].z;
    T lastZ = pZ;
    xyFront[pX] = pY;

    T area = (refX - pX) * (refY - pY);
    T volume = 0.0;

    // and then the rest of the points are processed
    for (py::ssize_t i = 1U, n = points.size(); i < n; i++) {
        if (points[i] == points[i - 1U]) {
            continue;  // fast skip duplicate
        }

        pX = points[i].x;
        pY = points[i].y;
        pZ = points[i].z;

        // find greatest q_x, such that q_x <= p_x
        auto nodeQ = xyFront.lower_bound(pX);
        if (nodeQ->first > pX) {
            --nodeQ;
        }
        const auto qY = nodeQ->second;

        if (!(pY < qY)) {  // p is by dominated q
            continue;
        }

        volume += area * (pZ - lastZ);
        lastZ = pZ;

        // remove dominated points and their area contributions
        auto prevX = pX;
        auto prevY = qY;
        auto nodeS = ++nodeQ;
        auto sX = nodeS->first;
        auto sY = nodeS->second;

        while (true) {
            area -= (sX - prevX) * (refY - prevY);
            if (pY > sY) {  // guaranteed by the sentinel point
                break;
            }
            prevX = sX;
            prevY = sY;
            // 'nodeS' points to a dominated point before calling erase,
            // and to that points successor afterwards.
            nodeS = xyFront.erase(nodeS);
            sX = nodeS->first;
            sY = nodeS->second;
        }

        // add the new point (here 's' is 't' in the paper)
        area += (sX - pX) * (refY - pY);
        xyFront[pX] = pY;
    }

    // last point's contribution to the volume
    volume += area * (refZ - lastZ);

    return volume;
}

}  // namespace hv3d

#endif  // ANGUILLA_HYPERVOLUME_HV3D_HPP
