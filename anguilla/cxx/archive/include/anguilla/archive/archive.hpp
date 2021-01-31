#pragma once

#ifndef ANGUILLA_ARCHIVE_HPP
#define ANGUILLA_ARCHIVE_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Boost Intrusive
#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>

// STL
#include <algorithm>
#include <cmath>
#include <iterator>
#include <list>
#include <map>
#include <optional>

// Anguilla
#include <anguilla/archive/individual.hpp>
#include <anguilla/archive/parameters.hpp>
#include <anguilla/archive/statistics.hpp>

/*
DESCRIPTION

A 2-D archive for use with UP-MO-CMA-ES (see [2016:mo-cma-es]).

REFERENCES

[2013:2d-archive]
I. Hupkens, & M. Emmerich (2013).
Logarithmic-Time Updates in SMS-EMOA and Hypervolume-Based Archiving.
In EVOLVE - A Bridge between Probability, Set Oriented Numerics, and Evolutionary Computation IV (pp. 155–169).
Springer International Publishing.

[2016:mo-cma-es]
O. Krause, T. Glasmachers, N. Hansen, & C. Igel (2016).
Unbounded Population MO-CMA-ES for the Bi-Objective BBOB Test Suite.
In GECCO'16 - Companion of Proceedings of the 2016 Genetic and Evolutionary Computation Conference (pp. 1177–1184).

[2017:moo-archive]
T. Glasmachers (2016). A Fast Incremental Archive for Multi-objective OptimizationCoRR, abs/1604.01169.
*/

namespace archive {

template <typename T>
struct Node : public boost::intrusive::avl_set_base_hook<boost::intrusive::optimize_size<true>> {
    explicit Node(const py::array_t<T> &point, const py::array_t<T> &fitness, T step_size = 1.0, T p_succ = 0.5) : individual(point, fitness, step_size, p_succ) {
        individual.containerPtr = this;
    }

    friend bool operator<(const Node &l, const Node &r) { return l.individual < r.individual; }

    Individual<T> individual;
};

template <typename T>
using BaseArchive = boost::intrusive::avl_set<Node<T>, boost::intrusive::compare<std::less<Node<T>>>>;

template <typename T>
class Archive;

template <typename T>
struct NodeDisposer {
    void operator()(Node<T> *instancePtr) {
        delete instancePtr;
    }
};

template <typename T>
class Archive {
    using NodeTraits = typename BaseArchive<T>::node_traits;
    using NodeIterator = typename BaseArchive<T>::iterator;
    using NodePointer = typename BaseArchive<T>::node_ptr;
    struct ArchiveIterator;

   public:
    explicit Archive(const Parameters<T> &parameters, const std::optional<py::array_t<T>> &reference = std::nullopt) : m_parameters(parameters), m_reference(reference) {
        constexpr auto max = std::numeric_limits<T>::max();
        constexpr auto lowest = std::numeric_limits<T>::lowest();
        Node<T> *ySentinel;
        Node<T> *xSentinel;
        const py::array_t<T> empty(0);
        if (reference.has_value()) {
            auto referenceR = reference->template unchecked<1>();
            const std::array<T, 2> xS = {referenceR(0), lowest};
            const std::array<T, 2> yS = {lowest, referenceR(1)};
            const py::array_t<T> xSA(2, xS.data());
            const py::array_t<T> ySA(2, yS.data());
            xSentinel = new Node<T>(empty, xSA);
            ySentinel = new Node<T>(empty, ySA);
        } else {
            const std::array<T, 2> xS = {max, lowest};
            const std::array<T, 2> yS = {lowest, max};
            const py::array_t<T> xSA(2, xS.data());
            const py::array_t<T> ySA(2, yS.data());
            xSentinel = new Node<T>(empty, xSA);
            ySentinel = new Node<T>(empty, ySA);
        }
        m_archive.insert(*ySentinel);
        m_archive.insert(*xSentinel);
    }

    ~Archive() {
        m_archive.clear_and_dispose(m_disposer);
    }

    [[nodiscard]] auto insert(const py::array_t<T> &point, const py::array_t<T> &fitness) {
        auto newNode = new Node<T>(point, fitness, m_parameters.initialStepSize, m_parameters.pTargetSucc);
        return insertInternal(newNode);
    }

    [[nodiscard]] Individual<T> *insertInternal(Node<T> *newNode) {
        // Based on the algorithm presented in [2013:2d-archive].

        m_statistics.insertAttempts++;
        const auto pX = newNode->individual.coord(0);
        const auto pY = newNode->individual.coord(1);
        // We seek the greatest 'qX' s.t. 'qX' =< 'pX'.
        // The interface of lower_bound returns an iterator that
        // points to the first 'qX' s.t. 'qX' >= 'pX'.
        // Otherwise, the iterator points to the end.
        // Sentinels guarantee that the following call succeds:
        auto nodeLeft = m_archive.lower_bound(*newNode);

        assert(nodeLeft != m_archive.end());
        assert(!(nodeLeft->individual.coord(0) < pX));
        if (nodeLeft->individual.coord(0) > pX) {
            --nodeLeft;
        }
        assert(!(nodeLeft->individual.coord(0) > pX));
        // Find if 'p' is dominated by 'q' or otherwise.
        if (!(nodeLeft->individual.coord(1) > pY)) {
            delete newNode;
            return nullptr;  // 'p' is dominated by 'q'
        }
        if (!(nodeLeft->individual.coord(0) < pX)) {  // qX == pX
            --nodeLeft;
        }

        // Find points dominated by p and delete them:
        auto nodeRight = nodeLeft;
        ++nodeRight;
        assert(nodeRight != m_archive.end());
        while (!(pY > nodeRight->individual.coord(1))) {
            ++nodeRight;
        }
        assert(nodeRight != m_archive.end());
        auto hintNode = m_archive.erase_and_dispose(std::next(nodeLeft), nodeRight, m_disposer);

        // Insert point
        const auto pContribution = (nodeRight->individual.coord(0) - pX) * (nodeLeft->individual.coord(1) - pY);
        newNode->individual.contribution = pContribution;
        newNode->individual.accContribution = 0.0;
        auto node = m_archive.insert(hintNode, *newNode);
        assert(static_cast<Node<T> *>(node.pointed_node()) == newNode);

        // Update contributions of left and right nodes
        const bool updateLeft = nodeLeft != m_archive.begin();
        if (updateLeft) {
            const auto prev = std::prev(nodeLeft);
            const auto lContrib = (pX - nodeLeft->individual.coord(0)) * (prev->individual.coord(1) - nodeLeft->individual.coord(1));
            nodeLeft->individual.contribution = lContrib;
        }
        const auto next = std::next(nodeRight);
        const bool updateRight = next != m_archive.end();
        if (updateRight) {
            const auto rContrib = (next->individual.coord(0) - nodeRight->individual.coord(0)) * (pY - nodeRight->individual.coord(1));
            nodeRight->individual.contribution = rContrib;
        }

        updateAccContributions();  // FIXME: O(n)

        if (!m_reference.has_value()) {
            // Always reset the extreme points to 'max'.
            constexpr auto max = std::numeric_limits<T>::max();
            rightExtremeIndividual()->contribution = max;
            leftExtremeIndividual()->contribution = max;
        }
        m_statistics.inserts++;
        return &(node->individual);
    }

    [[nodiscard]] std::size_t
    size() const {
        return (m_archive.size() - 2U);
    }

    [[nodiscard]] bool empty() const {
        return m_archive.size() == 2U;
    }

    [[nodiscard]] auto nbytes() const {
        return m_archive.size() * sizeof(Node<T>);
    }

    [[nodiscard]] auto reference() const {
        return m_reference;
    }

    [[nodiscard]] Statistics getStatistics() const {
        return m_statistics;
    }

    [[nodiscard]] constexpr auto isInteriorNode(NodePointer ptr) {
        return (ptr != leftExtreme().pointed_node()) &&
               (ptr != rightExtreme().pointed_node()) &&
               (ptr != leftSentinel().pointed_node()) &&
               (ptr != rightSentinel().pointed_node());
    }

    [[nodiscard]] constexpr auto isSentinelNode(NodePointer ptr) {
        return (ptr == leftSentinel().pointed_node()) ||
               (ptr == rightSentinel().pointed_node());
    }

    [[nodiscard]] constexpr auto isExtremeNode(NodePointer ptr) {
        return (ptr == leftExtreme().pointed_node()) ||
               (ptr == rightExtreme().pointed_node());
    }

    [[nodiscard]] constexpr auto rightSentinel() {
        return std::prev(m_archive.end());
    }

    [[nodiscard]] constexpr auto leftSentinel() {
        return m_archive.begin();
    }

    [[nodiscard]] constexpr auto leftExtreme() {
        return std::next(leftSentinel());
    }

    [[nodiscard]] constexpr auto rightExtreme() {
        return std::prev(rightSentinel());
    }

    [[nodiscard]] constexpr auto *leftExtremeIndividual() {
        return (size() != 0U) ? &(leftExtreme()->individual) : nullptr;
    }

    [[nodiscard]] constexpr auto *rightExtremeIndividual() {
        return (size() != 0U) ? &(rightExtreme()->individual) : nullptr;
    }

    [[nodiscard]] std::pair<Individual<T> *, Individual<T> *> nearest(Individual<T> *individualPtr) {
        auto *nodePtr = reinterpret_cast<NodePointer>(individualPtr->containerPtr);
        NodeIterator node;
        node = nodePtr;  // no suitable constructor is available, so we use assignment
        const auto left = std::prev(node);
        const auto right = std::next(node);
        const auto size = this->size();
        if (size >= 3U) {
            if (left == leftSentinel()) {
                return {&(right->individual), &(std::next(right)->individual)};
            }
            if (right == rightSentinel()) {
                return {&(std::prev(left)->individual), &(left->individual)};
            }
            return {&(left->individual), &(right->individual)};
        } else {
            if (size == 2U) {
                if (left == leftSentinel()) {
                    return {nullptr, &(right->individual)};
                }
                if (right == rightSentinel()) {
                    return {&(left->individual), nullptr};
                }
                return {&(left->individual), &(right->individual)};
            } else {
                return {nullptr, nullptr};
            }
        }
    }

    [[nodiscard]] Individual<T> *sampleExtreme(T p) {
        if (p > 1.0 || p < 0.0) {
            throw std::invalid_argument("p must be between zero and one");
        }
        if (p < 0.5) {
            return leftExtremeIndividual();
        }
        return rightExtremeIndividual();
    }

    // p ~ Uniform[0, 1]
    [[nodiscard]] Individual<T> *sampleInterior(T p) {
        if (p > 1.0 || p < 0.0) {
            throw std::invalid_argument("p must be between zero and one");
        }
        auto root = m_archive.root().pointed_node();
        T accContrib = static_cast<Node<T> *>(root)->individual.accContribution;
        auto current = std::next(m_archive.begin());
        T accP = 0.0;
        std::optional<Individual<T> *> result = std::nullopt;
        while (!result.has_value() || current == m_archive.end()) {
            if (!isExtremeNode(static_cast<Node<T> *>(current->individual.containerPtr))) {
                auto contrib = std::pow(current->individual.contribution, m_parameters.alpha);
                accP += contrib / accContrib;
                if (p < accP) {
                    result = &(current->individual);
                }
            }
            current = std::next(current);
        }
        return result.value_or(nullptr);
    }

    T updateAccContributions() {
        // FIXME: O(n). Update should be O(log n).
        // FIXME: recursion. Use iteration + stack.
        auto root = m_archive.root().pointed_node();
        if (root) {
            return _updateAccContributions(root);
        }
        return 0.0;
    }

    [[nodiscard]] T _updateAccContributions(NodePointer &ptr) {
        auto self = static_cast<Node<T> *>(ptr);
        if (!isExtremeNode(ptr)) {
            // We store the sum of contribution_i^alpha for i in the subtree.
            // A sentinel has 0 contribution always, so it doesn't affect the
            // accumulated value of its subtree.
            self->individual.accContribution = std::pow(self->individual.contribution, m_parameters.alpha);
        } else {
            // Don't consider the infinite contibution of extreme nodes.
            self->individual.accContribution = 0.0;
        }
        auto left = NodeTraits::get_left(ptr);
        if ((left != nullptr)) {
            self->individual.accContribution += _updateAccContributions(left);
        }
        auto right = NodeTraits::get_right(ptr);
        if ((right != nullptr)) {
            self->individual.accContribution += _updateAccContributions(right);
        }
        return self->individual.accContribution;
    }

    [[nodiscard]] auto begin() {
        return ArchiveIterator(std::forward<NodeIterator>(std::next(m_archive.begin())));
    }

    [[nodiscard]] auto end() {
        return ArchiveIterator(std::forward<NodeIterator>(std::prev(m_archive.end())));
    }

    void merge(Archive<T> &other) {
        auto current = std::next(other.m_archive.begin());  // left extreme
        auto end = std::prev(other.m_archive.end());        // right sentinel
        while (current != end) {
            auto nodePtr = current.pointed_node();
            auto tmp = current++;
            other.m_archive.erase(tmp);
            (void)insertInternal(static_cast<Node<T> *>(nodePtr));
        }
    }

   private:
    Statistics m_statistics;
    Parameters<T> m_parameters;
    std::optional<py::array_t<T>> m_reference;
    BaseArchive<T> m_archive;
    NodeDisposer<T> m_disposer;
};

template <typename T>
struct Archive<T>::ArchiveIterator {
    // Custom iterator implementation based on:
    // - https://internalpointers.com/post/writing-custom-iterators-modern-cpp
    // - https://en.cppreference.com/w/cpp/iterator/iterator_traits
    using NodeIterator = typename Archive<T>::NodeIterator;
    using difference_type = typename NodeIterator::difference_type;
#if __cplusplus >= 201703L
    using value_type = std::remove_cv_t<Individual<T>>;
#else
    using value_type = Individual<T>;
#endif
    using pointer = Individual<T> *;
    using reference = Individual<T> &;
    using iterator_category = typename NodeIterator::iterator_category;

    explicit ArchiveIterator(NodeIterator &&nodeIterator) : m_iterator(nodeIterator) {}

    reference operator*() const { return m_iterator->individual; }
    pointer operator->() { return &(m_iterator->individual); }

    auto &operator++() {
        ++m_iterator;
        return *this;
    }

    auto operator++(int) {
        auto tmp = *this;
        ++m_iterator;
        return tmp;
    }

    auto &operator--() {
        --m_iterator;
        return *this;
    }

    auto operator--(int) {
        auto tmp = *this;
        --m_iterator;
        return tmp;
    }

    friend bool operator==(const ArchiveIterator &l, const ArchiveIterator &r) { return l.m_iterator == r.m_iterator; };
    friend bool operator!=(const ArchiveIterator &l, const ArchiveIterator &r) { return l.m_iterator != r.m_iterator; };

   private:
    NodeIterator m_iterator;
};

}  // namespace archive
#endif  // ANGUILLA_ARCHIVE_HPP
