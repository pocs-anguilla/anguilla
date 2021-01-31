#pragma once

#ifndef ANGUILLA_INDIVIDUAL_HPP
#define ANGUILLA_INDIVIDUAL_HPP

// Pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace archive {

template <typename T>
struct Individual {
    explicit Individual(py::array_t<T> const &point, py::array_t<T> const &fitness, T step_size = 1.0, T p_succ = 0.5) : containerPtr(nullptr), contribution(0.0), accContribution(0.0), step_size(step_size), p_succ(p_succ), point(py::array_t<T>::ensure(point)), fitness(py::array_t<T>::ensure(fitness)), fitnessR(this->fitness.template unchecked<1>()), cov(nullptr) {
        // cov is initialized to be the identity matrix
        cov = new T[static_cast<std::size_t>(point.shape(0)) * static_cast<std::size_t>(point.shape(0))];
        const auto d = static_cast<std::size_t>(point.shape(0));
        const auto d_sq = d * d;
        for (auto i = 0U; i < d_sq; ++i) {
            cov[i] = 0.0;
        }
        for (auto i = 0U; i < d; ++i) {
            cov[i * d + i] = 1.0;
        }
    }
    ~Individual(){
        //delete[] cov;
    };

    friend bool operator<(const Individual &l, const Individual &r) {
        return l.coord(0) < r.coord(0);
    }

    friend bool operator>(const Individual &l, const Individual &r) {
        return l.coord(0) > r.coord(0);
    }

    friend bool operator==(const Individual &l, const Individual &r) {
        return l.coord(0) == r.coord(0);
    }

    inline T operator()(std::size_t index) const {
        return fitnessR(index);
    }

    inline T coord(std::size_t index) const {
        return fitnessR(index);
    }

    auto getCov() const {
        const auto d = point.shape(0);
        // Note: in order to avoid returning a copy we need to provide a
        // handle to the array constructor.
        // In this case we can use the 'point' data member.
        return py::array_t<T>({d, d}, cov, point);
    }

    // TODO: improve type safety.
    // Pointer to container (if applicable)
    void *containerPtr;

    // Individual contribution
    T contribution;
    // Accumulated contribution of the subtree
    T accContribution;
    // Step size
    T step_size;
    // Smooth probability of success
    T p_succ;
    // Search point
    const py::array_t<T> point;
    // Objective fitness
    const py::array_t<T> fitness;
    const py::detail::unchecked_reference<T, 1> fitnessR;
    // Covariance matrix
    T *cov;
};

}  // namespace archive

#endif  // ANGUILLA_INDIVIDUAL_HPP
