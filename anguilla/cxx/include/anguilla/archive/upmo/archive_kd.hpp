#pragma once

#ifndef ANGUILLA_UPMO_ARCHIVE_KD_HPP
#define ANGUILLA_UPMO_ARCHIVE_KD_HPP

// PyBind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Spatial
#include <spatial/point_multiset.hpp>

// Anguilla
#include <anguilla/archive/upmo/individual.hpp>
#include <anguilla/archive/upmo/parameters.hpp>

/*
DESCRIPTION

A k-D archive for use with UP-MO-CMA-ES (see [2016:mo-cma-es]).

NOTES

This is a work-in-progress version that will allow for implementing
UP-MO-CMA-ES for 3-D (or higher) objective space, although with a complexity
worse than that of the bi-objective version: O(log n).

REFERENCES

[2016:mo-cma-es]
O. Krause, T. Glasmachers, N. Hansen, & C. Igel (2016).
Unbounded Population MO-CMA-ES for the Bi-Objective BBOB Test Suite.
In GECCO'16 - Companion of Proceedings of the 2016 Genetic and Evolutionary
Computation Conference (pp. 1177–1184).

[2017:moo-archive]
T. Glasmachers (2016). A Fast Incremental Archive for Multi-objective
OptimizationCoRR, abs/1604.01169.
*/

namespace anguilla {
namespace upmo {

template <typename T>
using BaseArchiveKD = spatial::point_multiset<0, Individual<T>>;

template <typename T> class ArchiveKD {
  public:
    // FIXME: implement.
    struct ArchiveKDIterator;

    // FIXME: implement.
    explicit ArchiveKD(
        const Parameters<T>& parameters,
        const std::optional<py::array_t<T>>& reference = std::nullopt);

    ~ArchiveKD() = default;

    // FIXME: implement.
    [[nodiscard]] auto insert(const py::array_t<T>& point,
                              const py::array_t<T>& fitness);

    [[nodiscard]] std::size_t size() const { return (m_archive.size() - 2U); }

    [[nodiscard]] bool empty() const { return m_archive.size() == 2U; }

    // FIXME: implement.
    [[nodiscard]] auto nbytes() const;

    [[nodiscard]] auto reference() const { return m_reference; }

    // FIXME: implement.
    [[nodiscard]] Individual<T>* leftExterior();
    [[nodiscard]] Individual<T>* rightExterior();
    [[nodiscard]] std::pair<Individual<T>*, Individual<T>*>
    nearest(Individual<T>* individualPtr);
    [[nodiscard]] Individual<T>* sampleExterior(T p);
    [[nodiscard]] Individual<T>* sampleInterior(T p);

    // FIXME: implement.
    [[nodiscard]] auto begin();
    [[nodiscard]] auto end();
    [[nodiscard]] auto prev(ArchiveKDIterator& it);
    [[nodiscard]] auto next(ArchiveKDIterator& it);

    // FIXME: implement.
    void merge(ArchiveKD<T>& other);

  private:
    Parameters<T> m_parameters;
    std::optional<py::array_t<T>> m_reference;
    BaseArchiveKD<T> m_archive;
};

} // namespace upmo
} // namespace anguilla

#endif // ANGUILLA_UPMO_ARCHIVE_KD_HPP
