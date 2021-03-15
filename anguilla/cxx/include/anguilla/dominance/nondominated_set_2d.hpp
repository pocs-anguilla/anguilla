#pragma once

#ifndef ANGUILLA_NONDOMINATED_SET_2D_HPP
#define ANGUILLA_NONDOMINATED_SET_2D_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Boost Intrusive
#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>

// STL
#include <array>
#include <cstdint>

/*
DESCRIPTION

A non-dominated set of 2-D points.
Useful for merging independently generated non-dominated point sets.
*/

namespace anguilla {
namespace dominance {

template <typename T>
struct Node2D : public boost::intrusive::avl_set_base_hook<
                    boost::intrusive::optimize_size<true>> {
    Node2D(T x, T y) : x(x), y(y) {}

    friend bool operator<(const Node2D& l, const Node2D& r) {
        return l.x < r.x;
    }

    T x;
    T y;
};

template <typename T>
using BaseNonDominatedSet2D =
    boost::intrusive::avl_set<Node2D<T>,
                              boost::intrusive::compare<std::less<Node2D<T>>>>;

template <typename T> struct Node2DDisposer {
    void operator()(Node2D<T>* instancePtr) { delete instancePtr; }
};

template <typename T> class NonDominatedSet2D {
  public:
    NonDominatedSet2D() {
        constexpr auto max = std::numeric_limits<T>::max();
        constexpr auto lowest = std::numeric_limits<T>::lowest();
        auto ySentinel = new Node2D<T>(max, lowest);
        auto xSentinel = new Node2D<T>(lowest, max);
        m_set.insert(*ySentinel);
        m_set.insert(*xSentinel);
    }

    ~NonDominatedSet2D() { m_set.clear_and_dispose(m_disposer); }

    [[nodiscard]] auto insert(const py::array_t<T>& points) {
        auto pointsR = points.template unchecked<2>();
        assert(pointsR.shape(0) >= 0);
        const auto n = static_cast<std::size_t>(pointsR.shape(0));
        std::int64_t inserts = 0;
        for (std::size_t i = 0U; i < n; ++i) {
            const T x = pointsR(i, 0);
            const T y = pointsR(i, 1);
            auto newNode = new Node2D<T>(x, y);
            if (insertInternal(newNode)) {
                ++inserts;
            }
        }
        return inserts;
    }

    [[nodiscard]] bool insertInternal(Node2D<T>* newNode) {
        const auto pX = newNode->x;
        const auto pY = newNode->y;
        // We seek the greatest 'qX' s.t. 'qX' =< 'pX'.
        // The interface of lower_bound returns an iterator that
        // points to the first 'qX' s.t. 'qX' >= 'pX'.
        // Otherwise, the iterator points to the end.
        // Sentinels guarantee that the following call succeds:
        auto nodeLeft = m_set.lower_bound(*newNode);

        assert(nodeLeft != m_set.end());
        assert(!(nodeLeft->x < pX));
        if (nodeLeft->x > pX) {
            --nodeLeft;
        }
        assert(!(nodeLeft->x > pX));
        // Find if 'p' is dominated by 'q' or otherwise.
        if (!(nodeLeft->y > pY)) {
            delete newNode;
            return false; // 'p' is dominated by 'q'
        }
        if (!(nodeLeft->x < pX)) { // qX == pX
            --nodeLeft;
        }

        // Find points dominated by p and delete them:
        auto nodeRight = nodeLeft;
        ++nodeRight;
        assert(nodeRight != m_set.end());
        while (!(pY > nodeRight->y)) {
            ++nodeRight;
        }
        assert(nodeRight != m_set.end());
        auto hintNode =
            m_set.erase_and_dispose(std::next(nodeLeft), nodeRight, m_disposer);

        // Insert point
        m_set.insert(hintNode, *newNode);
        return true;
    }

    [[nodiscard]] std::size_t size() const { return (m_set.size() - 2U); }

    [[nodiscard]] bool empty() const { return m_set.size() == 2U; }

    [[nodiscard]] auto upperBound() const {
        constexpr auto lowest = std::numeric_limits<T>::lowest();
        std::array<T, 2> resultA = {lowest, lowest};
        auto current = std::next(m_set.begin()); // left extreme
        auto end = std::prev(m_set.end());       // right sentinel
        while (current != end) {
            if (resultA[0] < current->x) {
                resultA[0] = current->x;
            }
            if (resultA[1] < current->y) {
                resultA[1] = current->y;
            }
            ++current;
        }
        py::array_t<T> result(2, resultA.data());
        return result;
    }

    void merge(NonDominatedSet2D<T>& other) {
        auto current = std::next(other.m_set.begin()); // left extreme
        auto end = std::prev(other.m_set.end());       // right sentinel
        while (current != end) {
            auto nodePtr = current.pointed_node();
            auto tmp = current++;
            other.m_set.erase(tmp);
            (void)insertInternal(static_cast<Node2D<T>*>(nodePtr));
        }
    }

  private:
    BaseNonDominatedSet2D<T> m_set;
    Node2DDisposer<T> m_disposer;
};

} // namespace dominance
} // namespace anguilla
#endif // ANGUILLA_NONDOMINATED_SET_HPP
