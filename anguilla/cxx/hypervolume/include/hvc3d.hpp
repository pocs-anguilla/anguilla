#ifndef ANGUILLA_HYPERVOLUME_HVC3D_HPP
#define ANGUILLA_HYPERVOLUME_HVC3D_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <algorithm>
#include <list>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

#include "common.hpp"

namespace hvc3d {

static constexpr const char *docstring =
    "Calculate the exact hypervolume contributions for a set of 3-D points.";

template <typename T>
struct Box3D {
    Box3D(const Point3D<T> &lower, const Point3D<T> &upper)
        : l_x(lower.x), l_y(lower.y), l_z(lower.z), u_x(upper.x), u_y(upper.y), u_z(upper.z) {}

    inline T volume() const {
        return (u_x - l_x) * (u_y - l_y) * (u_z - l_z);
    }

    T l_x;
    T l_y;
    T l_z;
    T u_x;
    T u_y;
    T u_z;
};

// The sweeping structure is mantained in ascending order by x-coordinate.
template <typename T>
auto xComp = [](const Point3D<T> &lhs, const Point3D<T> &rhs) -> bool {
    return lhs.x < rhs.x;
};

template <typename T = double>
using RBTreeMap = std::map<Point3D<T>, py::ssize_t, decltype(xComp<T>)>;

template <typename T = double>
using BTreeMap = btree::map<Point3D<T>, py::ssize_t, decltype(xComp<T>)>;

template <typename T = double, class Map = BTreeMap<T>>
py::array_t<T> contributions(const Point3DList<T> &points, const Point3D<T> &refPoint) {
    const auto n = points.size();

    constexpr auto NaN = std::numeric_limits<T>::signaling_NaN();
    constexpr auto lowest = std::numeric_limits<T>::lowest();

    // Here we allocate memory for the contributions, initialized to zero,
    // plus an additional element for the sentinel nodes.
    auto contributionsPtr = new std::vector<T>(n + 1U);
    auto &contributions = *contributionsPtr;
    contributions[n] = NaN;

    py::capsule freeContributionsMemory(contributionsPtr, [](void *ptr) {
        auto concretePtr = static_cast<decltype(contributionsPtr)>(ptr);
        delete concretePtr;
    });

    // We work on a vector but will return a Numpy array.
    // It uses the vector's memory, which will be freed once the array
    // goes out of scope (handled by the py::capsule).
    const auto output = py::array_t<T>({n},
                                       {sizeof(T)},
                                       contributionsPtr->data(),
                                       freeContributionsMemory);

    if (n == 0U) {
        return output;
    }

    std::vector<std::pair<Point3D<T>, std::size_t>> input;
    input.reserve(n);
    {
        auto i = 0U;
        for (const auto &p : points) {
            input.emplace_back(p, i++);
        }
    }

    std::sort(input.begin(), input.end(),
              [](auto const &l, auto const &r) { return l.first.z < r.first.z; });

    // Create the sweeping structure.
    Map front(xComp<T>);

    // Add the xy-sentinels.
    const T refX = refPoint.x;
    const T refY = refPoint.y;
    const T refZ = refPoint.z;

    const auto xSentinel = Point3D<T>(refX, lowest, lowest);
    front.emplace(xSentinel, n);

    const auto ySentinel = Point3D<T>(lowest, refY, lowest);
    front.emplace(ySentinel, n);

    // Create the lists of boxes.
    std::vector<std::deque<Box3D<T>>> boxLists(n + 1U);

    // A working buffer for tracking dominated nodes.
    std::vector<std::size_t> dominatedIndices;

    // A working buffer for tracking repeated point mappings.
    std::vector<std::pair<std::size_t, std::size_t>> equalMappings;

    // Start by processing the first point.
    auto prevP = input[0U].first;
    auto prevIndex = input[0U].second;
    {
        auto box = Box3D<T>(prevP, refPoint);
        box.u_z = NaN;
        boxLists[prevIndex].push_back(std::move(box));
        front.insert(input[0U]);
    }

    // And process all the others.
    for (auto i = 1U; i < n; ++i) {
        const auto &p = input[i].first;
        const auto index = input[i].second;

        // Skip duplicate point and map its index to the previous one's to
        // output the correct value.
        if (prevP == p) {
            equalMappings.emplace_back(prevIndex, index);
            continue;
        } else {
            prevP = p;
            prevIndex = index;
        }

        // Note:
        //  - 'left' is also called 'q' in the paper
        //  - 'right' is also called 't' (or 's')

        // Find greatest q_x, such that q_x <= p_x:
        auto nodeLeft = front.lower_bound(p);
        if (nodeLeft->first.x > p.x) {
            --nodeLeft;
        }
        const auto left_index = nodeLeft->second;
        const auto &left = nodeLeft->first;

        if (!(p.y < left.y)) {  // p is by dominated left
            continue;
        }

        // (a) Find all points dominated by p, remove them
        // from the sweeping structure and add them to
        // the list of dominated nodes.
        // Also here we find the lowest t_x, such that p_x < t_x.
        auto nodeRight = nodeLeft;
        ++nodeRight;

        while (!(p.y > nodeRight->first.y)) {
            dominatedIndices.push_back(nodeRight->second);  // caution: reverse order
            ++nodeRight;
        }
        const auto right_index = nodeRight->second;
        const auto &right = nodeRight->first;

        // (b) Process "left" region (symmetric to the paper).
        {
            T volume = 0.0;
            auto &boxes = boxLists[left_index];
            while (!boxes.empty()) {
                auto &box = boxes.back();
                if (p.x < box.l_x) {
                    // This box is dominated at this z-level
                    // so it can completed and added to the
                    // volume contribution of the left neighbour.
                    box.u_z = p.z;
                    volume += box.volume();
                    boxes.pop_back();
                } else {
                    if (p.x < box.u_x) {
                        box.u_z = p.z;
                        volume += box.volume();
                        // Modify box to reflect the dominance
                        // of the left neighbour in this part
                        // of the L region.
                        box.u_x = p.x;
                        box.u_z = NaN;
                        box.l_z = p.z;
                        // Stop removing boxes
                        break;
                    } else {
                        break;
                    }
                }
            }
            contributions[left_index] += volume;
        }

        // (c) Process the dominated points.
        {
            auto rightX = nodeRight->first.x;

            for (auto dom_index_it = dominatedIndices.rbegin(), end = dominatedIndices.rend(); dom_index_it != end; ++dom_index_it) {
                auto dom_index = *dom_index_it;
                auto &d = points[dom_index];
                auto &boxes = boxLists[dom_index];
                // close boxes of dominated point 'd'
                for (auto &box : boxes) {
                    box.u_z = p.z;
                    contributions[dom_index] += box.volume();
                }
                // open box for current point 'p'
                {
                    auto box = Box3D<T>(p, d);
                    box.u_x = rightX;
                    box.u_z = NaN;
                    box.l_x = d.x;
                    boxLists[index].push_front(std::move(box));
                }
                rightX = d.x;
            }
            dominatedIndices.clear();
            {
                auto box = Box3D<T>(p, left);
                box.u_x = rightX;
                box.u_z = NaN;
                boxLists[index].push_front(std::move(box));
            }
        }

        // (d) Process "right" region (symmetric to the paper).
        {
            T volume = 0.0;
            auto rightX = right.x;
            auto &boxes = boxLists[right_index];
            while (!boxes.empty()) {
                auto &box = boxes.front();
                if (box.u_y > p.y) {
                    box.u_z = p.z;
                    volume += box.volume();
                    rightX = box.u_x;
                    boxes.pop_front();
                } else {
                    break;
                }
            }
            if (rightX > right.x) {
                auto box = Box3D<T>(right, p);
                box.u_x = rightX;
                box.u_z = NaN;
                box.l_z = p.z;
                boxLists[right_index].push_front(std::move(box));
            }
            contributions[right_index] += volume;
        }

        // Update the front.
        front.erase(std::next(nodeLeft), nodeRight);
        front.insert(input[i]);
    }

    // The paper uses a 'z sentinel' to close any remaining boxes.
    // Instead, here we do it as Shark's does.
    for (auto it = front.begin(), end = front.end(); it != end; ++it) {
        const auto index = it->second;
        auto &boxes = boxLists[index];
        for (auto &box : boxes) {
            box.u_z = refZ;
            contributions[index] += box.volume();
        }
    }

    // Finally, we handle duplicates by copying the corresponding
    // contributions.
    for (const auto &[orig, duplicate] : equalMappings) {
        contributions[duplicate] = contributions[orig];
    }

    return output;
}

}  // namespace hvc3d

#endif  // ANGUILLA_HYPERVOLUME_HVC3D_HPP
